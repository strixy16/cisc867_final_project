import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from regularizer import ElasticNetRegularizer, GroupLassoRegularizer, GroupSparseLassoRegularizer, L1Regularizer, L2Regularizer
from torch.utils.data.dataset import Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class GaussianNoise(nn.Module):
    def __init__(self, std=0.06):
        super().__init__()
        # self.noise = torch.zeros(1).cuda()
        self.std = std

    def forward(self, x):
        if not self.training: return x
        noise = torch.zeros(x.size()).cuda()
        noise.normal_(0,std=self.std)
        # self.noise.data.normal_(0, std=self.std)
        return x + noise


def get_accuracy(outs, inpts):
    inpts = inpts.to(device, dtype=torch.long)
    _, predict = torch.max(outs.data, 1)
    tot = inpts.size(0)
    corr = (predict == inpts).sum().item()
    acc = corr/tot
    return acc


def get_patient_acc(predict, in_pat, id_pat, prob_pat):
    u_id = np.unique(id_pat)

    corr = np.asarray(predict == in_pat)
    incorr = np.asarray(predict != in_pat)
    pat_acc_tot, pat_prob_tot, pat_in_tot = [], [], []
    for i in u_id:
        id_loc = np.asarray(np.where(id_pat == i)).squeeze()
        corr_pat = corr[id_loc].sum()
        incorr_pat = incorr[id_loc].sum()
        prob_score = prob_pat[id_loc].mean()
        in_score = in_pat[id_loc].mean()
        if corr_pat >= incorr_pat:
            pat_score = 1
        else:
            pat_score = 0
        pat_acc_tot = np.hstack((pat_acc_tot, pat_score))
        pat_prob_tot = np.hstack((pat_prob_tot, prob_score))
        pat_in_tot = np.hstack((pat_in_tot, in_score))
    pat_acc = 100 * np.asarray(pat_acc_tot).mean()
    return pat_acc, pat_prob_tot, pat_in_tot, u_id


def train(model, x, y, criter, optim, reg_type, lambda_reg):
    model.train()
    y_pred, p = model(x)
    optim.zero_grad()
    loss = criter(y_pred.squeeze(), y)
    if reg_type == 'EL':
        loss = ElasticNetRegularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)
    if reg_type == 'GL':
        loss = GroupLassoRegularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)
    if reg_type == 'SGL':
        loss = GroupSparseLassoRegularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)
    if reg_type == 'L1':
        loss = L1Regularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)
    if reg_type == 'L2':
        loss = L2Regularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)

    loss.backward()
    optim.step()
    return loss.item()


def valid(model, x_valid, y_valid, criter, reg_type, lambda_reg):
    model.eval()
    y_pred, p = model(x_valid)
    loss = criter(y_pred.squeeze(), y_valid)
    if reg_type == 'EL':
        loss = ElasticNetRegularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)
    if reg_type == 'GL':
        loss = GroupLassoRegularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)
    if reg_type == 'SGL':
        loss = GroupSparseLassoRegularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)
    if reg_type == 'L1':
        loss = L1Regularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)
    if reg_type == 'L2':
        loss = L2Regularizer(model=model, lambda_reg=lambda_reg).regularized_all_param(reg_loss_function=loss)

    return loss.item()
