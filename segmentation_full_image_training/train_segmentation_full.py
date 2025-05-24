import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preparation.data_loader import LungDataset
from torchvision import transforms

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU()
            )
        self.encoder1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder1 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.decoder1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        return torch.sigmoid(self.final(d2))


transform = transforms.Compose([transforms.ToTensor()])
dataset = LungDataset(r"D:\iti\Computer_Vision\project_data\train\images", r"D:\iti\Computer_Vision\project_data\train\masks", transform=transform, task="segmentation")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    model.train()
    total_loss = 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "unet_full.pth")
