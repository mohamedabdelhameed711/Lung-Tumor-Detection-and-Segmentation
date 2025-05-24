import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CroppedTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, detection_dir, transform=None):
        self.samples = []
        self.transform = transform

        for subject in os.listdir(image_dir):
            subject_path = os.path.join(image_dir, subject)
            for fname in os.listdir(subject_path):
                img_path = os.path.join(subject_path, fname)
                det_path = os.path.join(detection_dir, subject, fname.replace('.png', '.txt'))
                mask_path = os.path.join(mask_dir, subject, fname)

                if os.path.exists(det_path):
                    with open(det_path, 'r') as f:
                        for i, line in enumerate(f.readlines()):
                            xmin, ymin, xmax, ymax = map(int, line.strip().split(','))
                            self.samples.append((img_path, mask_path, (xmin, ymin, xmax, ymax)))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, box = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        xmin, ymin, xmax, ymax = box

        # Crop image and mask
        image = image.crop((xmin, ymin, xmax, ymax))
        mask = mask.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
