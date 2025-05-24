import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, detection_dir=None, transform=None, task="detection"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.detection_dir = detection_dir
        self.transform = transform
        self.task = task
        self.samples = []

        for subject in os.listdir(image_dir):
            subject_path = os.path.join(image_dir, subject)
            for fname in os.listdir(subject_path):
                image_path = os.path.join(subject_path, fname)
                mask_path = os.path.join(mask_dir, subject, fname) if mask_dir else None
                detection_path = os.path.join(detection_dir, subject, fname.replace('.png', '.txt')) if detection_dir else None
                self.samples.append((image_path, mask_path, detection_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path, detection_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.task == "segmentation":
            mask = Image.open(mask_path).convert("L")
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            return image, mask

        else:  # detection
            boxes = []
            if os.path.exists(detection_path):
                with open(detection_path) as f:
                    for line in f:
                        coords = list(map(int, line.strip().split(',')))  # ['134', '179', '136', '183'] -> [134, 179, 136, 183]
                        boxes.append(coords)

            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)  # Handle empty case


            if self.transform:
                image = self.transform(image)

            target = {}
            target = {
                "boxes": boxes,
                "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),  # assume all are tumors
                "image_id": torch.tensor([idx])
            }
            
            return image, target
