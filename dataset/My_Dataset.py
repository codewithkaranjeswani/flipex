import random
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class My_Dataset(Dataset):
	def __init__(self, root, img_dir):
		self.pil2tensor = transforms.ToTensor()
		self.new_size = (224, 224)
		self.resize = torchvision.transforms.Resize(self.new_size)
		df = pd.read_csv(os.path.join(root, 'train10.csv'), sep=',')
		self.img_dir = img_dir
		self.id_list = df['id'].to_list()
		self.fn_list = df['filename'].to_list()
		self.cat_list = df['category'].to_list()
		vertical_attribute_dict = np.load(os.path.join(root, "vertical_attributes.npy"),\
			allow_pickle=True).tolist()
		
		att_list = list(vertical_attribute_dict.keys())
		self.ndims = len(att_list)
		self.att_dict = {att_list[x] : x for x in range(len(att_list))}
		self.rev_dict = {x : att_list[x] for x in range(len(att_list))}

	def __getitem__(self, index):
		category_name = self.cat_list[index]
		ind = self.att_dict[category_name]
		target = torch.zeros(self.ndims, 1)
		target[ind] = 1
		img = Image.open(os.path.join(self.img_dir, self.fn_list[index]))
		img = self.resize(img)
		img = self.pil2tensor(img)
		return {"input": img, "target": target, "img_path": self.fn_list[index], \
			"id": self.id_list[index], "ind": ind, "category": self.rev_dict[ind]}

	def __len__(self):
		return len(self.fn_list)
