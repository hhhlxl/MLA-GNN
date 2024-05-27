import os
import logging
import numpy as np
import pandas as pd
import random
import pickle

import torch

# Env
from utils import *
from model_GAT import *
from options import parse_args
from test_model import test
from model_GAT import *


### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:0')
# 把这个改成1应该就不会报错了，但是15折交叉验证，剩下的数据都去哪了？如果我要换别的疾病数据集，怎样预处理数据？
num_splits = 1
results = []

### 2. Sets-Up Main Loop
for k in range(1, num_splits+1):
	print("*******************************************")
	print("************** SPLIT (%d/%d) **************" % (k, num_splits))
	print("*******************************************")

	# 这里的task给的是grad，没有给survival的预训练权重
	tr_features, tr_labels, te_features, te_labels, adj_matrix = load_csv_data(k, opt)
	# 加载预训练模型
	load_path = opt.model_dir + '/split' + str(k) + '_' + opt.task + '_' + str(
				opt.lin_input_dim) + 'd_all_' + str(opt.num_epochs) + 'epochs.pt'
	model_ckpt = torch.load(load_path, map_location=device)

	#### Loading Env
	model_state_dict = model_ckpt['model_state_dict']
	# hasattr(target, attr) 用于判断对象中是否含有某个属性，有则返回true.
	if hasattr(model_state_dict, '_metadata'):
		del model_state_dict._metadata

	# 从GAT网络构建模型
	model = GAT(opt=opt, input_dim=opt.input_dim, omic_dim=opt.omic_dim, label_dim=opt.label_dim,


				
				dropout=opt.dropout, alpha=opt.alpha).cuda()

	### multiple GPU
	# model = torch.nn.DataParallel(model)
	# torch.backends.cudnn.benchmark = True

	# 是否为实例？
	if isinstance(model, torch.nn.DataParallel): model = model.module

	print('Loading the model from %s' % load_path)
	model.load_state_dict(model_state_dict)


	### 3.2 Test the model.
	# te_features=features_all  te_fc_features=fc_features_all
	loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, te_features, te_fc_features = test(
		opt, model, te_features, te_labels, adj_matrix)
	# 又把两个特征和label合并起来..
	GAT_te_features_labels = np.concatenate((te_features, te_fc_features, te_labels), axis=1)

	# print("model preds:", list(np.argmax(pred_test[3], axis=1)))
	# print("ground truth:", pred_test[4])
	# print(te_labels[:, 2])

	# 保存特征和label?
	pd.DataFrame(GAT_te_features_labels).to_csv(
	    "./results/"+opt.task+"/GAT_features_"+str(opt.lin_input_dim)+"d_model/split"+str(k)+"_"+ opt.which_layer+"_GAT_te_features.csv")

	if opt.task == 'surv':
		print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		results.append(cindex_test)
	elif opt.task == 'grad':
		print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		results.append(grad_acc_test)

	# 又把预测结果拼接起来
	test_preds_labels = np.concatenate((pred_test[3], np.expand_dims(pred_test[4], axis=1)), axis=1)
	# (147, 4)
	print(test_preds_labels.shape)

	# 保存分级的预测结果
	pd.DataFrame(test_preds_labels, columns=["class1", "class2", "class3", "pred_class"]).to_csv(
		"./results/" + opt.task + "/preds/split" + str(k) + "_" + opt.which_layer + "_test_preds_labels.csv")
	# pickle.dump(pred_test, open(os.path.join(opt.results_dir, opt.task,
	# 		'preds/split%d_pred_test_%dd_%s_%depochs.pkl' % (k, opt.lin_input_dim, opt.which_layer, opt.num_epochs)), 'wb'))

# 0.7755102040816326
print('Split Results:', results)
# 0.7755102040816326
print("Average:", np.array(results).mean())
