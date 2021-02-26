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
from torchvision.models import resnet18, resnext101_32x8d
import torch.nn.functional as F
from torch.backends import cudnn
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
# parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
# parser.add_argument("--dataset_name", type=str, default="ixi_dataset_mri_pd_t2_split_randomly", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--use_gpu", type=int, default=1, help="cpu: 0, gpu: 1")
# parser.add_argument("--img_height", type=int, default=256, help="size of image height")
# parser.add_argument("--img_width", type=int, default=256, help="size of image width")
# parser.add_argument("--sample_interval", type=int, default=1, help="epochs after which we sample of images from generators")
# parser.add_argument("--checkpoint_interval", type=int, default=200, help="epochs between model checkpoints")
parser.add_argument("--gen", type=str, default="resnet", help="Selecting generator: UNet | AnamNet | ENet")
parser.add_argument("--in_ch", type=int, default=1, help="Considering neighbouring slices from input")
parser.add_argument("--lambda_pixel", type=float, default=2, help="lambda_pixel weight the reconstruction loss")
parser.add_argument("--lambda_vgg", type=float, default=0, help="lambda_vgg weight for perceptual loss")
opt = parser.parse_args()
# print(opt)

if opt.use_gpu and torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("Running on the GPU")
else:
	device = torch.device("cpu")
	print("Running on the CPU")

root = "../Sample_Data_Readme_and_other_docs"
vertical_attribute_dict = np.load(os.path.join(root, "vertical_attributes.npy"), allow_pickle=True).tolist()
no_of_classes = len(vertical_attribute_dict.keys())

model = resnext101_32x8d(pretrained=True)
# for param in model.parameters():
# 	param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
		nn.Dropout(0.5),
		nn.Linear(num_ftrs, no_of_classes)
)

model = model.to(device)

train_set = My_Dataset(root, "../train10_images")
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=8)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
cross_entropy = torch.nn.CrossEntropyLoss().to(device)


pbar_epoch = tqdm(total=len(range(opt.n_epochs)), desc='epochs', leave=False)
for epoch in range(1, opt.n_epochs+1):
	gt_index, pred_index = [], []
	running_loss = 0
	pbar_train = tqdm(total=len(train_loader), desc='train', leave=False)
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

		pbar_train.set_description("train: Loss: {0}, Acc: {1}".format(\
			round(loss.item(), 3), round(acc_batch, 3)))
		pbar_train.update(1)


	acc = np.mean(np.array(gt_index) == np.array(pred_index))
	# print("epoch: ", epoch," acc: ", acc, " loss: ", loss.cpu())
	pbar_epoch.set_description("epochs: Loss: {0}, Acc: {1}".format(\
			round(running_loss / opt.batch_size, 3), round(acc, 3)))
	pbar_epoch.update(1)

torch.save(model.state_dict(), "resnext101_32x8d_{0:02d}.pth".format(opt.n_epochs+1))