import torch
import os
from Attribute_Dataset import My_Dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnext101_32x8d, resnext50_32x4d
import torch
import torch.nn as nn

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("Running on the GPU")
else:
	device = torch.device("cpu")
	print("Running on the CPU")

model = resnext50_32x4d(pretrained=False)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)

