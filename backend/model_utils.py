import torch
import torch.nn as nn
from torchvision import models, transforms

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=5):
    # Use weights='DEFAULT' for the latest PyTorch versions
    model = models.resnet18(weights='DEFAULT') 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])