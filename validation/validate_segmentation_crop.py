from segmentation_cropped_training.train_cropped_unet import CroppedTumorDataset
from segmentation_cropped_training.train_cropped_unet import UNet_Cropped
from torch.utils.data import DataLoader
from utils import iou_score, dice_score, bbox_iou
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

val_dataset = CroppedTumorDataset(
    image_dir=r"D:\iti\Computer_Vision\project_data\val\images",
    mask_dir=r"D:\iti\Computer_Vision\project_data\val\masks",
    detection_dir=r"D:\iti\Computer_Vision\project_data\val\detections",
    transform=transform
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
model = UNet_Cropped().to("cuda")
model.load_state_dict(torch.load(r"D:\iti\Computer_Vision\project\segmentation_cropped_training\unet_cropped.pth"))
model.eval()

ious, dices = [], []

import matplotlib.pyplot as plt
import numpy as np

threshold = 0.5  

with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to("cuda"), masks.to("cuda")
        preds = model(images)

        # Compute metrics
        ious.append(iou_score(preds, masks))
        dices.append(dice_score(preds, masks))

        # Plot for each image in batch
        for i in range(images.size(0)):
            img_np = images[i].cpu().permute(1, 2, 0).numpy()  # CHW -> HWC
            gt_mask = masks[i].cpu().squeeze().numpy()
            pred_mask = (preds[i].cpu().squeeze().numpy() > threshold).astype(np.uint8)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img_np)
            axs[0].set_title("Input Image")
            axs[0].axis("off")

            axs[1].imshow(gt_mask, cmap='gray')
            axs[1].set_title("Ground Truth Mask")
            axs[1].axis("off")

            axs[2].imshow(pred_mask, cmap='gray')
            axs[2].set_title("Predicted Mask")
            axs[2].axis("off")

            plt.tight_layout()
            plt.show()

print(f"Cropped Segmentation - IOU: {sum(ious)/len(ious):.4f}, Dice: {sum(dices)/len(dices):.4f}")
