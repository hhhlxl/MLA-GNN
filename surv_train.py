import os
import logging
import numpy as np
import pandas as pd
import random
import pickle
import torch
# Env
from utils import *
from options import parse_args
from test_model import test
from my_model_GAT import *
from torch import optim
import torch.utils.data as Data

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:0')
# 把这个改成1应该就不会报错了，但是15折交叉验证，剩下的数据都去哪了？如果我要换别的疾病数据集，怎样预处理数据？
epochs = 5
results = []

# 准备数据
tr_features, tr_labels, adj_matrix = load_csv_data_my(opt)
train_dataset = Data.TensorDataset(tr_features, tr_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

# 从GAT网络构建模型
# 模型也需要修改，加入cox回归预测的部分，不要用classifier
model = GAT(opt=opt, input_dim=opt.input_dim, omic_dim=opt.omic_dim, label_dim=opt.label_dim,
            dropout=opt.dropout, alpha=opt.alpha).cuda()

# 是否为实例？
# if isinstance(model, torch.nn.DataParallel): model = model.module

# 设置损失函数和优化器
optimizer = define_optimizer(opt, model)
scheduler = define_scheduler(opt, optimizer)
loss_func = CoxLoss()


### 2. Sets-Up Main Loop
for k in range(1, epochs + 1):
    print("*******************************************")
    print("************** EPOCH (%d/%d) **************" % (k, epochs))
    print("*******************************************")

    model.train()

    # 定义了几个np数组，不知道要干嘛
    loss_train = 0
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])

    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        # 生存状态
        censor = batch_labels[:, 0]
        # 生存时间
        survtime = batch_labels[:, 1]
        censor_batch_labels = censor.cuda() if "surv" in opt.task else censor
        surv_batch_labels = survtime
        # 把数据和label输入模型,得到结果?但是这里输入的label是分级label,没有输入生存label
        # 输出：GAT提取的特征(GAT_features),全连接层的特征(fc_features),模型输出(out),梯度(gradients),不知道是啥(feature_importance)
        tr_preds = model(batch_features.cuda(), adj_matrix.cuda(), surv_batch_labels, opt)

        # 损失计算
        loss_cox = loss_func(surv_batch_labels, censor_batch_labels, tr_preds)
        loss_train += loss_cox.data.item()

        risk_pred_all = np.concatenate(
            (risk_pred_all, tr_preds.detach().cpu().numpy().reshape(-1)))  # Logging Information
        censor_all = np.concatenate(
            (censor_all, censor_batch_labels.detach().cpu().numpy().reshape(-1)))  # Logging Information
        survtime_all = np.concatenate(
            (survtime_all, surv_batch_labels.detach().cpu().numpy().reshape(-1)))  # Logging Information

        # 反向传播和优化
        optimizer.zero_grad()
        loss_cox.backward(retain_graph=True)
        optimizer.step()
    scheduler.step()

    model.eval()
    loss_train /= len(train_loader.dataset)
    cindex_train = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_train = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_train = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None

    print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    logging.info("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    results.append(cindex_train)

# 0.7755102040816326
print('Epoch Results:', results)
# 0.7755102040816326
print("Average:", np.array(results).mean())
