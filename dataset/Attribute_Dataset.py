import random
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import ast

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class My_Dataset(Dataset):
	def __init__(self, csv_file, img_dir, train = False):
		self.train  =train
		if self.train:
			some = '/home/rahulroy/Flip_ex/Sample_Data_Readme_and_other_docs'
			ano_path = os.path.join(some)
			self.img_dir = img_dir
			self.pil2tensor = transforms.ToTensor()
			self.new_size = (224, 224)
			self.resize = torchvision.transforms.Resize(self.new_size)

			# self.attribute_allowed_values = np.load(os.path.join('/home', *thing, \
			# 	"Sample_Data_Readme_and_other_docs", "Attribute_allowedvalues.npy"), allow_pickle=True).tolist()

			with open('/home/rahulroy/Flip_ex/Notebooks/attribute_values.json', 'r') as json_file:
				self.attribute_allowed_values = json.load(json_file)

			for i in self.attribute_allowed_values.keys():
				if 'none' in self.attribute_allowed_values[i]:
					self.attribute_allowed_values[i].remove('none')
				dct = {x: index for index,x in enumerate(self.attribute_allowed_values[i])}
				dct['none'] = len(dct)
				self.attribute_allowed_values[i] = dct

			df = pd.read_csv(os.path.join(csv_file), sep=',')
			self.attributes = df['attributes'].to_list()
			self.id_list = df['id'].to_list()
			self.fn_list = df['filename'].to_list()
			self.cat_list = df['category'].to_list()
			vertical_attribute_dict = np.load(\
				os.path.join(ano_path, "vertical_attributes.npy"), allow_pickle=True).tolist()
			
			att_list = sorted(vertical_attribute_dict.keys())
			self.ndims = len(att_list)
			self.att_dict = {att_list[x] : x for x in range(len(att_list))}
			self.rev_dict = {x : att_list[x] for x in range(len(att_list))}
		else:
			self.img_dir = img_dir
			self.pil2tensor = transforms.ToTensor()
			self.new_size = (224, 224)
			self.resize = torchvision.transforms.Resize(self.new_size)

			# self.attribute_allowed_values = np.load(os.path.join('/home', *thing, \
			# 	"Sample_Data_Readme_and_other_docs", "Attribute_allowedvalues.npy"), allow_pickle=True).tolist()

			# with open('/home/rahulroy/Flip_ex/Notebooks/attribute_values.json', 'r') as json_file:
			# 	self.attribute_allowed_values = json.load(json_file)

			# for i in self.attribute_allowed_values.keys():
			# 	if 'none' in self.attribute_allowed_values[i]:
			# 		self.attribute_allowed_values[i].remove('none')
			# 	dct = {x: index for index,x in enumerate(self.attribute_allowed_values[i])}
			# 	dct['none'] = len(dct)
			# 	self.attribute_allowed_values[i] = dct

			df = pd.read_csv(os.path.join(csv_file), sep=',')
			# self.attributes = df['attributes'].to_list()
			# self.id_list = df['id'].to_list()
			self.fn_list = df['filename'].to_list()
			# self.cat_list = df['category'].to_list()
			# vertical_attribute_dict = np.load(\
			# 	os.path.join(ano_path, "vertical_attributes.npy"), allow_pickle=True).tolist()
			
			# att_list = sorted(vertical_attribute_dict.keys())
			# self.ndims = len(att_list)
			# self.att_dict = {att_list[x] : x for x in range(len(att_list))}
			# self.rev_dict = {x : att_list[x] for x in range(len(att_list))}

	def __getitem__(self, index):
		if self.train:
			dct = ast.literal_eval(self.attributes[index])
			label_dict = {}
			for key in self.attribute_allowed_values.keys():
				label = torch.zeros(size=(len(self.attribute_allowed_values[key]),1))
				if key in dct.keys():
					att_list = dct[key]
					index_list = []
					for x in att_list:
						# print(x)
						label[self.attribute_allowed_values[key][x]] = 1
				else:
					label[self.attribute_allowed_values[key]['none']] = 1
				label_dict[key] = label

			category_name = self.cat_list[index]
			ind = self.att_dict[category_name]
			target = torch.zeros(self.ndims, 1)
			target[ind] = 1
			img = Image.open(os.path.join(self.img_dir, self.fn_list[index]))
			img = self.resize(img)
			img = self.pil2tensor(img)
			return {"input": img, "target": target, "img_fn": self.fn_list[index], \
				"id": self.id_list[index], "ind": ind, "category": self.rev_dict[ind], \
				"label_dict": label_dict}

		else:
			img = Image.open(os.path.join(self.img_dir, self.fn_list[index]))
			img = self.resize(img)
			img = self.pil2tensor(img)
			return {"input": img, "img_fn": self.fn_list[index]}

	def __len__(self):
		return len(self.fn_list)
