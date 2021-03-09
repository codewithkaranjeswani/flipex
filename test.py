import argparse
import os, glob
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from dataset.My_Dataset import My_Dataset

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnext101_32x8d, resnext50_32x4d
import torch.nn.functional as F
from torch.backends import cudnn

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="../Sample_Data_Readme_and_other_docs", help="location of dataset")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--use_gpu", type=int, default=1, help="cpu: 0, gpu: 1")
parser.add_argument("--ins", type=str, default="resnext50_32x4d_05.pth", help="Files in testset")
opt = parser.parse_args()
# print(opt)

if opt.use_gpu and torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("Running on the GPU")
else:
	device = torch.device("cpu")
	print("Running on the CPU")

root = opt.data_file
vertical_attribute_dict = np.load(os.path.join(root, "vertical_attributes.npy"), allow_pickle=True).tolist()
no_of_classes = len(vertical_attribute_dict.keys())

model = resnext50_32x4d(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, no_of_classes)
model = model.to(device)

model.load_state_dict(torch.load(opt.ins, map_location=device))
model.eval()

test_set = My_Dataset(root, "../train10_images")
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, \
  num_workers=opt.n_cpu, pin_memory=True, persistent_workers=True)

cross_entropy = torch.nn.CrossEntropyLoss().to(device)

gt_index, pred_index = [], []
running_loss = 0
pbar_test = tqdm(total=len(test_loader), desc='test', leave=False)
for i, batch in enumerate(test_loader):
  with torch.no_grad():
    inp = Variable(batch["input"].type(torch.FloatTensor)).to(device)
    target_ind = Variable(batch["ind"].type(torch.LongTensor)).to(device)
    # model.train()
    pred = model(inp)
    loss = cross_entropy(pred, target_ind)
    running_loss += loss.item()
    _, pred_ind = torch.max(pred, 1)

    gt_index += list(batch["ind"].detach().cpu().numpy())
    pred_index += list(pred_ind.detach().cpu().numpy())

    acc_batch = np.mean(np.array(gt_index) == np.array(pred_index))

    pbar_test.set_description("test : Loss: {0:6.3f}, Acc: {1:6.3f}".format(\
      round(loss.item(), 3), round(acc_batch, 3)))
    pbar_test.update(1)

acc = np.mean(np.array(gt_index) == np.array(pred_index))
# print("epoch: ", epoch," acc: ", acc, " loss: ", loss.cpu())
print("epochs: Loss: {0:6.3f}, Acc: {1:6.3f}".format(\
    round(running_loss / len(test_loader), 3), round(acc, 3)))
