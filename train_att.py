import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json

from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset.Attribute_Dataset import My_Dataset
from models.Net import Net

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnext101_32x8d, resnext50_32x4d
import torch.nn.functional as F
from torch.backends import cudnn
from utils.metrics import get_acc, get_output_df, mlabelacuracy
import wandb

wandb.init(project="attribute-prediction", entity="karanjeswani")

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)

parser = argparse.ArgumentParser()
parser.add_argument("--attr_dir", type=str, default="logs/current_run_att/models", help="Files in testset")
parser.add_argument("--ins", type=str, default="logs/current_run_att/resnext50_32x4d_not_pretrained_10_att.pth", help="Files in testset")
parser.add_argument("--n_epochs", type=int, default=7, help="number of epochs of training")
parser.add_argument("--wd", type=float, default=0, help="weight decay")
# parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--info_dir", type=str, default="../Sample_Data_Readme_and_other_docs", help="location of dataset")
parser.add_argument("--data_dir", type=str, default="../Data", help="location of dataset")
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
	device = torch.device("cuda:1")
	print("Running on the GPU")
else:
	device = torch.device("cpu")
	print("Running on the CPU")

train_set = My_Dataset(os.path.join(opt.data_dir, "train_split_74.76.csv"), \
	os.path.join(opt.data_dir, "train_split_74.76_images"), train=True)
valid_set = My_Dataset(os.path.join(opt.data_dir, "valid_split_25.24.csv"), \
	os.path.join(opt.data_dir, "valid_split_25.24_images"), train=True)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, \
	num_workers=opt.n_cpu, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=True, \
	num_workers=opt.n_cpu, pin_memory=True)

with open('/home/rahulroy/Flip_ex/Notebooks/attribute_values.json', 'r') as json_file:
	attribute_allowed_values = json.load(json_file)

model = resnext50_32x4d(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 26)
model = nn.Sequential(*list(model.children())[:-1]) # put this after next line
model.load_state_dict(torch.load(opt.ins, map_location=device))
# for params in model.parameters():
# 	params.requries_grad = False
model = model.to(device)

train_from_scratch = True
model_dict = {}
if train_from_scratch:
	for one in attribute_allowed_values.keys():
		n_classes = len(attribute_allowed_values[one])
		model_dict[one] = Net(n_classes).to(device)
else:
	for one in attribute_allowed_values.keys():
		n_classes = len(attribute_allowed_values[one])
		model_dict[one] = Net(n_classes).to(device)
		fil_name = os.path.join(opt.attr_dir,'model_{0:02d}_{1:s}.pth').format(10,one)
		model_dict[one].load_state_dict(torch.load(fil_name, map_location=device))
		model_dict[one].eval()
# wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)
BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss().to(device)

current_run_path = "logs/current_run_att_3"
if os.path.exists(current_run_path):
	new_name = "logs/run_att_train_archived_" + datetime.now().strftime('%y%m%d-%H%M%S')
	os.rename(current_run_path, new_name)
os.makedirs(current_run_path)
os.makedirs(os.path.join(current_run_path, 'csv'))
f = open("logs/current_train_run/csv/loss_train_{0}.csv".format(opt.n_epochs), "wt")
g = open("logs/current_train_run/csv/loss_valid_{0}.csv".format(opt.n_epochs), "wt")
f.write("step,train_acc,train_loss\n")
g.write("step,valid_acc,valid_loss\n")

pbar_epoch = tqdm(total=len(range(opt.n_epochs)), desc='epochs', position=0, leave=False)
for epoch in range(1, opt.n_epochs+1):
	### Training
	model.train()
	for one_key, one_model in model_dict.items():
		one_model.train()
	running_loss_train = 0
	running_acc_train = 0
	pbar_train = tqdm(total=len(train_loader), desc='Train ', position=1, leave=False)
	for i, batch in enumerate(train_loader):
		gt_index, pred_index = [], []
		inp = Variable(batch["input"].type(torch.FloatTensor)).to(device)
		img_emb = model(inp)
		loss = 0
		out = {}
		for one_key, one_model in model_dict.items():
			tar = Variable(batch["label_dict"][one_key].type(torch.FloatTensor)).to(device)
			out[one_key] = one_model(img_emb)
			loss += BCEWithLogitsLoss(out[one_key].squeeze(), tar.squeeze())

		optimizer.zero_grad()
		running_loss_train += loss.item()
		loss.backward()
		optimizer.step()
		with torch.no_grad():
			acc_dict = get_acc(out, batch["label_dict"], device)

		acc_batch = acc_dict['overall_acc']
		running_acc_train += acc_batch
		if (i%3 == 0):
			wandb.log({"Train Acc": round(acc_batch, 3), "Train Loss": round(loss.item(), 3), "train_step": (epoch-1)*110 + i//4})
			f.write("{0},{1},{2}\n".format((epoch-1)*len(valid_loader) + i//3, round(acc_batch, 3), round(loss.item(), 3)))
			f.flush()
		pbar_train.set_description("Train : Loss: {0:6.3f}, Acc: {1:6.3f}".format(\
			round(loss.item(), 3), round(acc_batch, 3)))
		pbar_train.update(1)

	### Validation
	model.eval()
	for one_key, one_model in model_dict.items():
		one_model.eval()
	running_loss_valid = 0
	running_acc_valid = 0
	pbar_valid = tqdm(total=len(valid_loader), desc='Valid ', position=2, leave=False)
	for i, batch in enumerate(valid_loader):
		with torch.no_grad():
			inp = Variable(batch["input"].type(torch.FloatTensor)).to(device)
			img_emb = model(inp)
			loss = 0
			out = {}
			for one_key, one_model in model_dict.items():
				tar = Variable(batch["label_dict"][one_key].type(torch.FloatTensor)).to(device)
				out[one_key] = one_model(img_emb)
				loss += BCEWithLogitsLoss(out[one_key].squeeze(), tar.squeeze())

			# optimizer.zero_grad()
			running_loss_valid += loss.item()
			# loss.backward()
			# optimizer.step()
			acc_dict = get_acc(out, batch["label_dict"], device)

			acc_batch = acc_dict['overall_acc']
			running_acc_valid += acc_batch
			wandb.log({"Val Acc": round(acc_batch, 3), "Val Loss": round(loss.item(), 3), "valid_step": (epoch-1)*110 + i})
			g.write("{0},{1},{2}\n".format((epoch-1)*len(valid_loader) + i, round(acc_batch, 3), round(loss.item(), 3)))
			g.flush()
			pbar_valid.set_description("Valid : Loss: {0:6.3f}, Acc: {1:6.3f}".format(\
				round(loss.item(), 3), round(acc_batch, 3)))
			pbar_valid.update(1)

	### Metrics
	avg_train_acc = running_acc_train / len(train_loader)
	avg_valid_acc = running_acc_valid / len(valid_loader)
	avg_train_loss = running_loss_train / len(train_loader)
	avg_valid_loss = running_loss_valid / len(valid_loader)

	scheduler.step(avg_train_loss)
	wandb.log({"Avg Train Acc": round(avg_train_acc, 3), "Avg Valid Acc": round(avg_valid_acc, 3),\
		"Avg Train Loss": round(avg_train_loss, 3), "Avg Valid Loss": round(avg_valid_loss, 3), "epoch": epoch})
	pbar_epoch.update(1)
	flag = os.path.isdir(os.path.join(current_run_path, 'models_{0:02d}'.format(epoch)))
	if not flag:
		os.makedirs(os.path.join(current_run_path, 'models_{0:02d}'.format(epoch)), exist_ok=False)
	for one_key, one_model in model_dict.items():
		fil_name = os.path.join(current_run_path, 'models_{0:02d}'.format(epoch),'model_{0:02d}_{1:s}.pth')\
			.format(opt.n_epochs, one_key)
		torch.save(one_model.state_dict(), fil_name)

	torch.save(model.state_dict(),  os.path.join(current_run_path,"resnext50_32x4d_pretrained_{0:02d}_att.pth".format(epoch)))

g.close()
f.close()

os.makedirs(os.path.join(current_run_path, 'models_end'), exist_ok=False)
for one_key, one_model in model_dict.items():
	fil_name = os.path.join(current_run_path, 'models_end','model_{0:02d}_{1:s}.pth')\
		.format(opt.n_epochs, one_key)
	torch.save(one_model.state_dict(), fil_name)

torch.save(model.state_dict(),  os.path.join(current_run_path,"resnext50_32x4d_pretrained_{0:02d}_att.pth".format(opt.n_epochs)))
