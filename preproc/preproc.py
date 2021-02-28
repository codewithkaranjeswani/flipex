root = "../Sample_Data_Readme_and_other_docs"
train_set = My_Dataset(root, "../train10_images")
train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)

category_name = self.cat_list[index]
ind = self.att_dict[category_name]
target = torch.zeros(self.ndims, 1)
target[ind] = 1
img = Image.open(os.path.join(self.img_dir, self.fn_list[index]))
newsize = (224, 224)
img = img.resize(newsize)
