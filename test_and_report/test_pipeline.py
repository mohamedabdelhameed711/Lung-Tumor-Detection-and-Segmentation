import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import os
from model_loader import load_detection_model, load_segmentation_model
from report_generator import generate_report
from validation.utils import iou_score, dice_score, bbox_iou

def run_pipeline(image_path, mode="cropped"):
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to("cuda")

    det_model = load_detection_model()
    seg_model = load_segmentation_model(mode)

    detections = det_model(image_tensor)[0]
    boxes = detections['boxes'].detach().cpu()
    scores = detections['scores'].detach().cpu()

    valid_boxes = [box.int().tolist() for i, box in enumerate(boxes) if scores[i] > 0.5]

    fragments = []
    for i, box in enumerate(valid_boxes):
        cropped_img = image_pil.crop(box)
        input_tensor = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])(cropped_img).unsqueeze(0).to("cuda")

        mask_pred = seg_model(input_tensor).squeeze().detach().cpu().numpy()
        fragments.append((box, mask_pred))

    generate_report(
        image_path=image_path,
        detections=valid_boxes,
        masks=fragments,
        output_path="/kaggle/working/tumor_report.pdf"
    )
    print('Report Generated!!')

if __name__ == "__main__":
    run_pipeline("/kaggle/input/lungtumordetectionandsegmentationdata/val/images/Subject_57/216.png", mode="cropped")
