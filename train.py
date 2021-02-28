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
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
# parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--data_file", type=str, default="../Sample_Data_Readme_and_other_docs", help="location of dataset")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--use_gpu", type=int, default=1, help="cpu: 0, gpu: 1")
# parser.add_argument("--img_height", type=int, default=256, help="size of image height")
# parser.add_argument("--img_width", type=int, default=256, help="size of image width")
# parser.add_argument("--sample_interval", type=int, default=1, help="epochs after which we sample of images from generators")
# parser.add_argument("--checkpoint_interval", type=int, default=200, help="epochs between model checkpoints")
opt = parser.parse_args()
# print(opt)

if opt.use_gpu and torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("Running on the GPU")
else:
	device = torch.device("cpu")
	print("Running on the CPU")

data_root = opt.data_file
vertical_attribute_dict = np.load(os.path.join(data_root, "vertical_attributes.npy"), allow_pickle=True).tolist()
no_of_classes = len(vertical_attribute_dict.keys())

model = resnext50_32x4d(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
		nn.Dropout(0.5),
		nn.Linear(num_ftrs, no_of_classes)
)
model = model.to(device)

train_set = My_Dataset(data_root, "../train10_images")
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
cross_entropy = torch.nn.CrossEntropyLoss().to(device)

pbar_epoch = tqdm(total=len(range(opt.n_epochs)), desc='epochs', position=0, leave=False)
for epoch in range(1, opt.n_epochs+1):
	gt_index, pred_index = [], []
	running_loss = 0
	pbar_train = tqdm(total=len(train_loader), desc='train', position=1, leave=False)
	for i, batch in enumerate(train_loader):
		inp = Variable(batch["input"].type(torch.FloatTensor)).to(device)
		target_ind = Variable(batch["ind"].type(torch.LongTensor)).to(device)
		# model.train()
		pred = model(inp)
		optimizer.zero_grad()
		loss = cross_entropy(pred, target_ind)
		running_loss += loss.item()
		loss.backward()
		optimizer.step()
		_, pred_ind = torch.max(pred, 1)

		gt_index += list(batch["ind"].detach().cpu().numpy())
		pred_index += list(pred_ind.detach().cpu().numpy())

		acc_batch = np.mean(np.array(gt_index) == np.array(pred_index))

		pbar_train.set_description("train : Loss: {0:6.3f}, Acc: {1:6.3f}".format(\
			round(loss.item(), 3), round(acc_batch, 3)))
		pbar_train.update(1)

	acc = np.mean(np.array(gt_index) == np.array(pred_index))
	# print("epoch: ", epoch," acc: ", acc, " loss: ", loss.cpu())
	pbar_epoch.set_description("epochs: Loss: {0:6.3f}, Acc: {1:6.3f}".format(\
			round(running_loss / opt.batch_size, 3), round(acc, 3)))
	pbar_epoch.update(1)

torch.save(model.state_dict(), "resnext50_32x4d_{0:02d}.pth".format(opt.n_epochs))