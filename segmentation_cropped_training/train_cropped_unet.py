import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from .cropped_dataset import CroppedTumorDataset

class UNet_Cropped(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        return torch.sigmoid(self.final(d2))

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = CroppedTumorDataset(
        image_dir=r"D:\iti\Computer_Vision\project_data\train\images",
        mask_dir=r"D:\iti\Computer_Vision\project_data\train\masks",
        detection_dir=r"D:\iti\Computer_Vision\project_data\train\detections",
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet_Cropped().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(50):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Batch Loss": loss.item()})

        print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "unet_cropped.pth")

