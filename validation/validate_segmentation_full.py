import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from utils import iou_score, dice_score, bbox_iou
from segmentation_full_image_training.train_segmentation_full import UNet 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
model.load_state_dict(torch.load(r"D:\iti\Computer_Vision\project\segmentation_full_image_training\unet_full.pth", map_location=device))
model.eval()

# Image and mask transforms
transform_img = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((128, 128), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float())  # binarize mask
])

# Evaluation paths
val_img = r"D:\iti\Computer_Vision\project_data\val\images"
val_mask = r"D:\iti\Computer_Vision\project_data\val\masks"

ious, dices = [], []
threshold = 0.5

with torch.no_grad():
    for subject in os.listdir(val_img):
        for fname in os.listdir(os.path.join(val_img, subject)):
            img_path = os.path.join(val_img, subject, fname)
            mask_path = os.path.join(val_mask, subject, fname)

            img = transform_img(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            gt = transform_mask(Image.open(mask_path).convert("L")).unsqueeze(0).to(device)

            pred = model(img)
            iou = iou_score(pred, gt)
            dice = dice_score(pred, gt)
            ious.append(iou)
            dices.append(dice)

            # Visualization
            if iou > 0 or dice > 0:  # only plot relevant ones
                pred_mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)
                gt_mask = gt.squeeze().cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(Image.open(img_path).convert("RGB"))
                axs[0].set_title("Original Image")
                axs[1].imshow(gt_mask, cmap='gray')
                axs[1].set_title("Ground Truth Mask")
                axs[2].imshow(pred_mask, cmap='gray')
                axs[2].set_title(f"Predicted Mask\nIoU: {iou:.3f}, Dice: {dice:.3f}")
                for ax in axs: ax.axis("off")
                plt.tight_layout()
                plt.show()

print(f"\nFull-image Segmentation Results:")
print(f"Average IoU: {sum(ious)/len(ious):.4f}")
print(f"Average Dice: {sum(dices)/len(dices):.4f}")
