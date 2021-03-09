import torch

def mlabelacuracy(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    intersection = torch.logical_and(y_true,y_pred)
    union = torch.logical_or(y_true,y_pred)
    num = torch.sum(intersection,axis=-1)
    den = torch.sum(union,axis=-1)
    iou = torch.where(torch.equal(num,0)&torch.equal(den,0),0,num/den)
    return iou
 
def get_acc(output_dict, tar_dict):
    acc_dict = {}
    acc_list = []
    for key in output_dict.keys():
        acc_list.append(mlabelacuracy(tar_dict[key].squeeze(), output_dict[key]))
        acc_dict[key] = acc_list[-1]
    acc_dict['overall_acc'] = torch.mean(torch.array(acc_list))
    return acc_dict
 
def get_output_df(output_dict, att_dict, df_list):
    predict_dict = []
    for key in output_dict.keys():
        _, index = torch.max(output_dict[key], 1)
        att_list = att_dict[key][index]
        if 'none' not in att_list:
            predict_dict[key] = att_list
        else:
             continue
    df_list.append(str(predict_dict))