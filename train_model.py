import numpy as np
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as Data
from torch import optim
from sklearn.model_selection import StratifiedKFold

from utils import *
from model_GAT import *



def train(opt, model, tr_features, tr_labels, adj_matrix):

    model.train()

    train_dataset = Data.TensorDataset(tr_features, tr_labels)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=False)

    # 定义了几个np数组，不知道要干嘛
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_train, grad_acc_train = 0, 0

    #
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        # 生存状态
        censor = batch_labels[:, 0]
        # 生存时间
        survtime = batch_labels[:, 1]
        # 分级
        grade = batch_labels[:, 2]
        # 如果是生存分析任务，则将生存状态转到cuda上，否则就不转
        censor_batch_labels = censor.cuda() if "surv" in opt.task else censor
        surv_batch_labels = survtime
        # print(surv_batch_labels)
        grad_batch_labels = grade.cuda() if "grad" in opt.task else grade
        # 把数据和label输入模型,得到结果?但是这里输入的label是分级label,没有输入生存label
        # 输出：GAT提取的特征(GAT_features),全连接层的特征(fc_features),模型输出(out),梯度(gradients),不知道是啥(feature_importance)
        tr_preds = model(
            batch_features.cuda(), adj_matrix.cuda(), grad_batch_labels, opt)

        # print("surv_batch_labels:", surv_batch_labels)
        # print("te_preds:", te_preds)


        # 如果是生存任务，则计算生存损失，否则为0
        loss_cox = CoxLoss(surv_batch_labels, censor_batch_labels, tr_preds) if opt.task == "surv" else 0
        # 这什么鬼东西
        loss_reg = define_reg(model)
        # 又来个交叉熵损失
        loss_func = nn.CrossEntropyLoss()
        # 如果是分级任务，就用交叉熵损失
        grad_loss = loss_func(tr_preds, grad_batch_labels) if opt.task == "grad" else 0
        # 反向传播
        grad_loss.backward()
        # 更新权重


        # 真实值？
        gt_all = np.concatenate((gt_all, grad_batch_labels.detach().cpu().numpy().reshape(-1)))  # Logging Information

        if opt.task == "surv":
            risk_pred_all = np.concatenate((risk_pred_all, tr_preds.detach().cpu().numpy().reshape(-1)))  # Logging Information
            censor_all = np.concatenate((censor_all, censor_batch_labels.detach().cpu().numpy().reshape(-1)))  # Logging Information
            survtime_all = np.concatenate((survtime_all, surv_batch_labels.detach().cpu().numpy().reshape(-1)))  # Logging Information

        elif opt.task == "grad":
            pred = tr_preds.argmax(dim=1, keepdim=True)
            grad_acc_train += pred.eq(grad_batch_labels.view_as(pred)).sum().item()
            probs_np = tr_preds.detach().cpu().numpy()
            probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)  # Logging Information

    # print(survtime_all)
    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_train /= len(train_loader.dataset)
    cindex_train = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_train = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_train = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None
    grad_acc_train = grad_acc_train / len(train_loader.dataset) if opt.task == 'grad' else None
    pred_train = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

    return loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train