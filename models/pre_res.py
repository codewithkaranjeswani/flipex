import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d

model = resnext50_32x4d(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 26)
)

x = torch.randn(5,3,224,224)

y = model(x)

print(y.shape)

# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)