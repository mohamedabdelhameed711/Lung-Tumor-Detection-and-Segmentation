from PIL import Image, ImageDraw
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_report(image_path, detections, masks, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Lung Tumor Detection Report", ln=1, align="C")
    pdf.cell(200, 10, txt=f"Input File: {os.path.basename(image_path)}", ln=1)

    if not detections:
        pdf.cell(200, 10, txt="Tumor: Not Detected", ln=1)
    else:
        pdf.cell(200, 10, txt=f"Tumor: Detected ({len(detections)} fragment(s))", ln=1)
        for i, (box, mask) in enumerate(masks):
            x0, y0, x1, y1 = box
            draw.rectangle(box, outline="red", width=2)
            area = np.sum(mask > 0.5)
            pdf.cell(200, 10, txt=f"Fragment {i+1} - Location: ({x0},{y0}) to ({x1},{y1}), Size: {area} px²", ln=1)

    vis_path = "/kaggle/working/temp_vis.png"
    image.save(vis_path)

    pdf.image(vis_path, x=10, y=None, w=pdf.w - 20)
    os.remove(vis_path)
    pdf.output(output_path)
