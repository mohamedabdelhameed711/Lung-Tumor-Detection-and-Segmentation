import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
from segmentation_full_image_training.train_segmentation_full import UNet as UNetFull
from segmentation_cropped_training.train_cropped_unet import UNet_Cropped



def load_detection_model(path=r"D:\iti\Computer_Vision\project\object_detection_training\FasterRCNN_Detector.pth"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2)
    model.load_state_dict(torch.load(path))
    model.eval().to("cuda")
    return model

def load_segmentation_model(mode="cropped", path=r"D:\iti\Computer_Vision\project\segmentation_cropped_training\unet_cropped.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if mode == "cropped":
        model = UNet_Cropped()
    else:
        model = UNetFull()
    
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
