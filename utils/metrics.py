import torch
import numpy as np

def mlabelacuracy(y_true, y_pred, dev):
	y_pred[y_pred >  0.5] = 1
	y_pred[y_pred <= 0.5] = 0
	intersection = np.logical_and(y_true,y_pred)
	union = np.logical_or(y_true,y_pred)
	num = np.sum(intersection,axis=-1)
	den = np.sum(union,axis=-1)
	iou = np.where(np.equal(num,0)&np.equal(den,0),0,num/den)
	return iou

def get_acc(output_dict, tar_dict, dev):
	acc_dict = {}
	acc_list = []
	for key in output_dict.keys():
		acc_list.append(mlabelacuracy(tar_dict[key].squeeze().cpu().numpy(),\
			output_dict[key].cpu().numpy(), dev))
		acc_dict[key] = acc_list[-1]
	acc_dict['overall_acc'] = np.mean(np.array(acc_list))
	return acc_dict

def get_output_df(output_dict, att_dict, df_list, index):
	predict_dict = {}
	for key in output_dict.keys():
		y_pred = output_dict[key].cpu().numpy()
		# print(y_pred.shape)
		y_pred[y_pred >  0.95] = 1
		y_pred[y_pred <= 0.95] = 0
		# print(y_pred[index,:].shape)
		ind = list(np.where(y_pred[index,:] == 1)[0])
		# print(ind)
		att_list = [att_dict[key][i] for i in ind]
		# att_list = att_dict[key][ind]
		# print(att_list)
		if len(att_list) == 0:
			continue
		if 'none' not in att_list:
			predict_dict[key] = att_list
		else:
			att_list = att_list.remove('none')
			if att_list is None:
				continue
			else:
				predict_dict[key] = att_list
	# df_list.append(str(predict_dict))
	return str(predict_dict)