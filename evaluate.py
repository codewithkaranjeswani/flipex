import numpy as np
import pandas as pd
import json
import ast
import os

def mlabelacuracy(y_true, y_pred):
  # print(y_true)
  # print(y_pred)
  intersection = np.logical_and(y_true,y_pred)
  # print(intersection)
  union = np.logical_or(y_true,y_pred)
  # print(union)
  num = np.sum(intersection,axis=-1)
  # print(num)
  den = np.sum(union,axis=-1)
  # print(den)
  iou = np.where(np.equal(num,0)&np.equal(den,0),0,num/den)
  return iou

df_pred = pd.read_csv('predictions_1000_95thres.csv', sep=',')
df_pred.sort_values(by=['filename'], inplace=True)

df_true = pd.read_csv('/home/rahulroy/Flip_ex/Image_data/PRED_sample_1000.csv', sep=',')
df_true.sort_values(by=['filename'], inplace=True)


print(df_true['filename'].tolist()==df_pred['filename'].tolist())
# print(df_true.head(5))
# print(df_pred.head(5))

with open('/home/rahulroy/Flip_ex/Notebooks/attribute_values_0.json', 'r') as json_file:
	attribute_allowed_values = json.load(json_file)

for i in attribute_allowed_values.keys():
  if 'none' in attribute_allowed_values[i]:
    attribute_allowed_values[i].remove('none')
  dct = {x: index for index,x in enumerate(attribute_allowed_values[i])}
  dct['none'] = len(dct)
  attribute_allowed_values[i] = dct

df = df_pred
acc_each = []; str_each = []
for pred, gt in zip(df_pred['predictions'].tolist(), df_true['predictions'].tolist()):
  pred_att_dict = ast.literal_eval(pred)
  gt_att_dict = ast.literal_eval(gt)
  running_acc = 0
  for one_key, one_att in pred_att_dict.items():
    for gt_key, gt_att in gt_att_dict.items():
      if gt_key == one_key:
        dim = len(attribute_allowed_values[gt_key])
        gt = np.zeros(shape=(dim,))
        pr = np.zeros(shape=(dim,))
        for one in gt_att:
          indy = attribute_allowed_values[gt_key][one]
          gt[indy] = 1
        for one in one_att:
          indy = attribute_allowed_values[one_key][one]
          pr[indy] = 1
        acc = mlabelacuracy(gt, pr)
        running_acc += acc
        # print(gt_key, one_att, gt_att, acc)
        # print(len(pred_att_dict))
  accu = round(np.float64((1.0 * running_acc)) / np.float64(len(pred_att_dict)), 8)
  avg_acc = "{0:0.8f}".format(accu)
  # print(avg_acc)
  str_each.append(avg_acc)
  acc_each.append(accu)

print(np.mean( np.array(acc_each) ) )

df['Acc'] = str_each
df.to_csv("accuracy_1000.csv", sep=',', index=False, header=True)
