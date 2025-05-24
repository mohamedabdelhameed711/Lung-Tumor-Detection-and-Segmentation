
# 🫁 Lung Tumor Detection & Segmentation 

This project is a web-based application built with **Dash** that enables users to upload lung CT scan images, detect tumors, and view a segmented report in PDF format.

## 🖼️ Live Demo

![App Demo](assets/demo.gif)

## 🚀 Features

- Upload CT scan images using drag-and-drop or file selection.
- Automatic tumor detection using image processing and ML model.
- Visualize detection output (e.g., bounding boxes).
- Segmented tumor report generated and rendered in a PDF viewer.
- Option to download the generated report.

## 🧠 Technologies Used

- 🐍 Python
- 📊 Dash
- 🖼️ OpenCV for image processing
- 📄 FPDF for PDF generation

## 📦 Installation

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/lung-tumor-segmentation-dash.git
   cd lung-tumor-segmentation-dash
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install Required Libraries

```bash
pip install -r requirements.txt
```

## 🧾 `requirements.txt` Example

```text
dash
dash-bootstrap-components
torch
torchvision
Pillow
fpdf
numpy
```

> ✨ You can update the file with additional model dependencies if needed (e.g., PyTorch, TensorFlow).

## 🧪 Running the App

```bash
python app.py
```

Then navigate to `http://127.0.0.1:8050/` in your browser.

## 📁 Project Structure

```
LUNG-TUMOR-DETECTION-AND-SEGMENTATION/
│
├── app.py                             # Dash app entry point
├── README.md                          # Project documentation
├── requirements.txt                   # All required packages
│
├── data_preparation/
│   └── data_loader.py                # Data loading and preprocessing
│
├── object_detection_training/
│   └── train_detector.py             # Training script for object detection model
│
├── segmentation_cropped_training/
│   ├── cropped_dataset.py            # Dataset and preprocessing for cropped segmentation
│   └── train_cropped_unet.py         # Training cropped segmentation model
│
├── segmentation_full_image_training/
│   └── train_segmentation_full.py    # Training full image segmentation model
│
├── test_and_report/
│   ├── model_loader.py               # Load trained models
│   ├── report_generator.py           # Generate analysis report as PDF
│   └── test_pipeline.py              # Full testing and inference pipeline
│
├── validation/
│   ├── utils.py                      # Utility functions
│   ├── validate_detection.py         # Validate object detection model
│   ├── validate_segmentation_crop.py # Validate cropped segmentation
│   └── validate_segmentation_full.py # Validate full image segmentation
├── assets/
│   ├── demo.gif

```

## 🖼️ Example Output

- Tumor fragments identified and highlighted.
- Bounding boxes overlaid on the original scan.
- PDF report with:
  - Fragment count
  - Tumor size
  - Coordinate locations
- Embedded viewer for instant preview.

---

## 📥 PDF Report

Each report includes:
- 🧩 Number of detected tumor regions
- 🔲 Bounding box coordinates (x, y, width, height)
- 📏 Approximate size (in pixels or mm)
- 📤 Downloadable via the app

---

## 🙌 Contributing

1. Fork the project.
2. Create your branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---