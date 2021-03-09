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
# import wandb

# wandb.init(project="attribute-prediction", entity="karanjeswani")

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="../../rahulroy/Flip_ex/Full_Data/all_images", help="location of dataset")
parser.add_argument("--cate", type=str, default="logs/current_run_att_2/resnext50_32x4d_pretrained_08_att.pth", help="Files in testset")
parser.add_argument("--attr_dir", type=str, default="logs/current_run_att_2/models_03", help="Files in testset")
parser.add_argument("--info_dir", type=str, default="../../rahulroy/Flip_ex/Sample_Data_Readme_and_other_docs", help="location of dataset")
parser.add_argument("--data_dir", type=str, default="../../rahulroy/Flip_ex/Image_data/", help="location of dataset")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--use_gpu", type=int, default=1, help="cpu: 0, gpu: 1")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
opt = parser.parse_args()
# print(opt)

if opt.use_gpu and torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("Running on the GPU")
else:
	device = torch.device("cpu")
	print("Running on the CPU")

root = opt.info_dir
vertical_attribute_dict = np.load(os.path.join(root, "vertical_attributes.npy"), allow_pickle=True).tolist()
no_of_classes = len(vertical_attribute_dict.keys())

model = resnext50_32x4d(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, no_of_classes)
model = nn.Sequential(*list(model.children())[:-1])
model.load_state_dict(torch.load(opt.cate, map_location=device))
model = model.to(device)
model.eval()

with open('/home/rahulroy/Flip_ex/Notebooks/attribute_values.json', 'r') as json_file:
	attribute_allowed_values = json.load(json_file)

model_dict = {}
for one in attribute_allowed_values.keys():
	n_classes = len(attribute_allowed_values[one])
	model_dict[one] = Net(n_classes).to(device)
	fil_name = os.path.join(opt.attr_dir,'model_{0:02d}_{1:s}.pth').format(8,one)
	model_dict[one].load_state_dict(torch.load(fil_name, map_location=device))
	model_dict[one].eval()

test_set = My_Dataset('/home/rahulroy/Flip_ex/Image_data/PRED_sample_1000.csv', \
	opt.data_file, False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True, \
  num_workers=opt.n_cpu, pin_memory=True, persistent_workers=True)

with open('/home/rahulroy/Flip_ex/Notebooks/attribute_values.json', 'r') as json_file:
	attribute_allowed_values = json.load(json_file)

BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss().to(device)

current_run_path = "logs/current_test_run_1000"
if os.path.exists(current_run_path):
	new_name = "logs/run_test_archived_" + datetime.now().strftime('%y%m%d-%H%M%S')
	os.rename(current_run_path, new_name)
os.makedirs(current_run_path)
os.makedirs(os.path.join(current_run_path, 'csv'))
# g = open("logs/current_test_run/csv/loss_test.csv", "wt")
# g.write("valid_acc,valid_loss\n")

### Test
model.eval()
final_df = []
running_loss_valid = 0
running_acc_valid = 0
pbar_valid = tqdm(total=len(test_loader), desc='Test ', position=0, leave=False)
for i, batch in enumerate(test_loader):
	with torch.no_grad():
		inp = Variable(batch["input"].type(torch.FloatTensor)).to(device)
		img_emb = model(inp)
		# loss = 0
		out = {}
		for one_key, one_model in model_dict.items():
			# tar = Variable(batch["label_dict"][one_key].type(torch.FloatTensor)).to(device)
			out[one_key] = one_model(img_emb)
			# loss += BCEWithLogitsLoss(out[one_key].squeeze(), tar.squeeze())

		# optimizer.zero_grad()
		# running_loss_train += loss.item()
		# loss.backward()
		# optimizer.step()
		# acc_dict = get_acc(out, batch["label_dict"], device)

		# acc_batch = acc_dict['overall_acc']
		# running_acc_valid += acc_batch
		for index in range(opt.batch_size):
			df = []
			df.append(batch['img_fn'][index])
			som = get_output_df(out, attribute_allowed_values, df, index)
			df.append(som)
			final_df.append(df)
			# print(final_df)
			# print(som)
		# wandb.log({"Val Acc": round(acc_batch, 3), "Val Loss": round(loss.item(), 3), "valid_step": (epoch-1)*110 + i})
		# g.write("{1},{2}\n".format(round(acc_batch, 3), round(loss.item(), 3)))
		# g.flush()
		# pbar_valid.set_description("Valid : Loss: {0:6.3f}, Acc: {1:6.3f}".format(\
		# 	round(loss.item(), 3), round(acc_batch, 3)))
		pbar_valid.update(1)
final_df = pd.DataFrame(final_df, columns = ['filename', 'predictions'])
final_df.to_csv('predictions_1000_95thres.csv', header=True, index=False)
	### Metrics
	# avg_train_acc = running_acc_train / len(train_loader)
	# avg_valid_acc = running_acc_valid / len(valid_loader)
	# avg_train_loss = running_loss_train / len(train_loader)
	# avg_valid_loss = running_loss_valid / len(valid_loader)

# g.close()
