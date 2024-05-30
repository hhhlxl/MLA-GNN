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
from torch import optim
import torch.utils.data as Data

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:0')
# 把这个改成1应该就不会报错了，但是15折交叉验证，剩下的数据都去哪了？如果我要换别的疾病数据集，怎样预处理数据？
epochs = 5
results = []

# 准备数据
tr_features, tr_labels, te_features, te_labels, adj_matrix = load_csv_data(1, opt)
train_dataset = Data.TensorDataset(tr_features, tr_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

# 从GAT网络构建模型
model = GAT(opt=opt, input_dim=opt.input_dim, omic_dim=opt.omic_dim, label_dim=opt.label_dim,
            dropout=opt.dropout, alpha=opt.alpha).cuda()

# 是否为实例？
# if isinstance(model, torch.nn.DataParallel): model = model.module

# 设置损失函数和优化器
optimizer = define_optimizer(opt, model)
scheduler = define_scheduler(opt, optimizer)
loss_func = nn.CrossEntropyLoss()


### 2. Sets-Up Main Loop
for k in range(1, epochs + 1):
    print("*******************************************")
    print("************** EPOCH (%d/%d) **************" % (k, epochs))
    print("*******************************************")

    model.train()

    # 定义了几个np数组，不知道要干嘛
    probs_all, gt_all = None, np.array([])
    loss_train, grad_acc_train = 0, 0

    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        # 分级
        grade = batch_labels[:, 2]
        # 如果是生存分析任务，则将生存状态转到cuda上，否则就不转
        grad_batch_labels = grade.cuda() if "grad" in opt.task else grade
        # 把数据和label输入模型,得到结果?但是这里输入的label是分级label,没有输入生存label
        # 输出：GAT提取的特征(GAT_features),全连接层的特征(fc_features),模型输出(out),梯度(gradients),不知道是啥(feature_importance)
        tr_preds = model(batch_features.cuda(), adj_matrix.cuda(), grad_batch_labels, opt)

        grad_loss = loss_func(tr_preds, grad_batch_labels)
        loss_train += grad_loss.data.item()

        pred = tr_preds.argmax(dim=1, keepdim=True)

        grad_acc_train += pred.eq(grad_batch_labels.view_as(pred)).sum().item()
        print("pred:", pred)
        print("label:", grad_batch_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        grad_loss.backward(retain_graph=True)
        optimizer.step()
    scheduler.step()

    model.eval()
    loss_train /= len(train_loader.dataset)
    grad_acc_train = grad_acc_train / len(train_loader.dataset)

    print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
    logging.info("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
    results.append(grad_acc_train)


# 0.7755102040816326
print('Epoch Results:', results)
# 0.7755102040816326
print("Average:", np.array(results).mean())
