import base64
import io
import os
import uuid
import tempfile
from pathlib import Path

import dash
from dash import html, dcc, Output, Input, State, ctx
import dash_bootstrap_components as dbc
import torch
import torchvision
from PIL import Image, ImageDraw
from fpdf import FPDF
from segmentation_cropped_training.train_cropped_unet import UNet_Cropped

# ---------- 1. Load models once ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2)
detector.load_state_dict(torch.load(
    r"D:\iti\Computer_Vision\project\object_detection_training\FasterRCNN_Detector.pth",
    map_location=device))
detector.to(device).eval()

segmenter = UNet_Cropped()
segmenter.load_state_dict(torch.load(
    r"D:\iti\Computer_Vision\project\segmentation_cropped_training\unet_cropped.pth",
    map_location=device, weights_only=True))
segmenter.to(device).eval()

# ---------- 2. Dash App ----------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    [
        html.H3(
            "Lung-Tumor Detection & Segmentation",
            style={
                "textAlign": "center",
                "backgroundColor": "#007BFF",
                "color": "white",
                "padding": "15px",
                "borderRadius": "8px",
                "marginTop": "10px"
            }
        ),
        dcc.Upload(
            id="uploader",
            children=html.Div(["Drag & Drop or ", html.A("Select Image")]),
            style={
                "width": "100%", "height": "60px", "lineHeight": "60px",
                "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                "textAlign": "center", "margin": "10px",
            },
            accept="image/*",
        ),
        html.Img(id="preview", style={"maxWidth": "300px", "display": "block", "margin": "10px auto"}),
        dbc.Button("Analyze", id="btn-analyze", color="primary", disabled=True),
        html.Div(id="analysis-output", className="mt-3"),
        html.Hr(),
        html.Iframe(
            id="pdf-frame",
            style={
                "width": "100%",
                "height": "90vh",        
                "minHeight": "600px",    
                "border": "none",
                "marginBottom": "20px",
            },
        ),
        html.Br(),
        dbc.Button("Download PDF", id="download-btn", color="success", disabled=True),
        dcc.Download(id="download-pdf"),
    ],
    fluid=True,
)

# ---------- 3. Helpers ----------
def run_pipeline(pil_img):
    transform = torchvision.transforms.ToTensor()
    tensor_img = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        det_out = detector(tensor_img)[0]
    keep = det_out["scores"] > 0.5
    boxes = det_out["boxes"][keep].cpu().int().tolist()

    vis = pil_img.copy()
    draw = ImageDraw.Draw(vis)
    fragments = []
    for box in boxes:
        draw.rectangle(box, outline="red", width=3)
        crop = pil_img.crop(box)
        crop_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            mask_pred = segmenter(crop_tensor).squeeze().cpu().numpy()
        area = (mask_pred > 0.5).sum()
        fragments.append({"box": box, "area": int(area)})

    return vis, fragments

def make_pdf(original_name, annotated_img, fragments):
    tmp_pdf = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Report for {original_name}", ln=1)
    if fragments:
        pdf.cell(0, 10, f"Tumor detected: {len(fragments)} fragment(s)", ln=1)
        for i, fr in enumerate(fragments, 1):
            pdf.cell(0, 8, f"Fragment {i}: bbox={fr['box']}, size(px)={fr['area']}", ln=1)
    else:
        pdf.cell(0, 10, "No tumor detected.", ln=1)

    img_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.png"
    annotated_img.save(img_path)
    pdf.image(str(img_path), x=10, y=None, w=180)
    pdf.output(tmp_pdf)
    img_path.unlink()
    return tmp_pdf

# ---------- 4. Callbacks ----------

@app.callback(
    Output("preview", "src"),
    Output("btn-analyze", "disabled"),
    Input("uploader", "contents"),
    prevent_initial_call=True,
)
def update_preview(content):
    if content is None:
        return dash.no_update, True
    return content, False

@app.callback(
    Output("analysis-output", "children"),
    Output("pdf-frame", "srcDoc"),
    Output("download-btn", "disabled"),
    Input("btn-analyze", "n_clicks"),
    State("uploader", "contents"),
    State("uploader", "filename"),
    prevent_initial_call=True,
)
def analyze(n_clicks, content, filename):
    if not content:
        return "Upload an image first.", "", True

    header, b64 = content.split(",")
    pil_img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    annotated, frags = run_pipeline(pil_img)
    pdf_path = make_pdf(filename, annotated, frags)

    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()

    pdf_src = f"data:application/pdf;base64,{pdf_b64}"
    pdf_html = f"""
    <!DOCTYPE html>
    <html style="margin:0;height:100%;">
    <body style="margin:0;height:100%;">
        <!-- make the PDF fill the whole iframe -->
        <embed src="{pdf_src}"
            type="application/pdf"
            style="position:absolute;top:0;left:0;width:100%;height:100%;" />
    </body>
    </html>
    """

    summary = f"✅ Analysis complete. {len(frags)} fragment(s) detected." if frags else "✅ No tumor detected."
    app.server.config.update(PDF_PATH=str(pdf_path))
    return summary, pdf_html, False

@app.callback(
    Output("download-pdf", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_pdf(n):
    pdf_path = app.server.config.get("PDF_PATH")
    if pdf_path and os.path.exists(pdf_path):
        return dcc.send_file(pdf_path, filename="Lung_Tumor_Report.pdf")
    return dash.no_update

# ---------- 5. Run ----------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
