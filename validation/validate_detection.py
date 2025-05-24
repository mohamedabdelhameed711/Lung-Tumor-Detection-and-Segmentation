import os
import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import iou_score, dice_score, bbox_iou

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(r"D:\iti\Computer_Vision\project\object_detection_training\FasterRCNN_Detector.pth", map_location=device))
model.to(device).eval()


val_path = r"D:\iti\Computer_Vision\project_data\val\images"
gt_path = r"D:\iti\Computer_Vision\project_data\val\detections"

transform = transforms.ToTensor()
ious = []
score_thresh = 0.5  

with torch.no_grad():
    for subject in os.listdir(val_path):
        subject_folder = os.path.join(val_path, subject)
        for fname in os.listdir(subject_folder):
            img_path = os.path.join(subject_folder, fname)
            det_path = os.path.join(gt_path, subject, fname.replace('.png', '.txt'))

            # Load image
            image_pil = Image.open(img_path).convert("RGB")
            image_tensor = transform(image_pil).unsqueeze(0).to(device)

            # Model prediction
            output = model(image_tensor)[0]
            pred_boxes = output['boxes'][output['scores'] > score_thresh].cpu().numpy()

            # Load GT boxes
            if os.path.exists(det_path):
                with open(det_path) as f:
                    gt_boxes = [list(map(int, line.strip().replace(',', ' ').split())) for line in f.readlines()]
            else:
                gt_boxes = []

            # Calculate IOU
            for pred in pred_boxes:
                if len(gt_boxes) == 0:
                    continue
                pred = pred.astype(int).tolist()
                ious.append(max(bbox_iou(pred, gt) for gt in gt_boxes))

            # Visualization 
            if len(pred_boxes) > 0 or len(gt_boxes) > 0:
                vis_image = image_pil.copy()
                draw = ImageDraw.Draw(vis_image)

                # Draw predicted boxes (Red)
                for box in pred_boxes:
                    box = box.astype(int).tolist()
                    draw.rectangle(box, outline='red', width=3)

                # Draw GT boxes (Green)
                for box in gt_boxes:
                    draw.rectangle(box, outline='green', width=3)

                # Show image
                plt.figure(figsize=(6, 6))
                plt.imshow(vis_image)
                plt.title(f"{fname} - Green: GT | Red: Predicted")
                plt.axis("off")
                plt.show()


if len(ious) > 0:
    print(f"Detection IOU Avg: {sum(ious)/len(ious):.4f}")
else:
    print("No valid IOU values computed.")
