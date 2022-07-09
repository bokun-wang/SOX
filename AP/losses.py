import torch
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
from utils import t_sigmoid, t_exp

sigmoidf = nn.Sigmoid()

class SOAPLOSS(nn.Module):
    def __init__(self, threshold, data_length):
        '''
        :param threshold: margin for squred hinge loss
        '''
        super(SOAPLOSS, self).__init__()
        self.u_all =  torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.u_pos =  torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.threshold = threshold

    def forward(self,f_ps, f_ns, index_s, gamma):
        f_ps = f_ps.view(-1)
        f_ns = f_ns.view(-1)

        vec_dat = torch.cat((f_ps, f_ns), 0)
        mat_data = vec_dat.repeat(len(f_ps), 1)

        f_ps = f_ps.view(-1, 1)

        neg_mask = torch.ones_like(mat_data)
        neg_mask[:, 0:f_ps.size(0)] = 0

        pos_mask = torch.zeros_like(mat_data)
        pos_mask[:, 0:f_ps.size(0)] = 1

        # 3*1 - 3*64 ==> 3*64

        neg_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * neg_mask
        pos_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * pos_mask
        loss = pos_loss + neg_loss

        if f_ps.size(0) == 1:
            self.u_pos[index_s] = (1 - gamma) * self.u_pos[index_s] + gamma * (pos_loss.mean())
            self.u_all[index_s] = (1 - gamma) * self.u_all[index_s] + gamma * (loss.mean())
        else:
            self.u_all[index_s] = (1 - gamma) * self.u_all[index_s] + gamma * (loss.mean(1, keepdim=True))
            self.u_pos[index_s] = (1 - gamma) * self.u_pos[index_s] + gamma * (pos_loss.mean(1, keepdim=True))
        p = (self.u_pos[index_s] - (self.u_all[index_s]) * pos_mask) / (self.u_all[index_s] ** 2)
        p.detach_()
        loss = torch.mean(p * loss)

        return loss

class MOAPV2LOSS(nn.Module):
    def __init__(self, threshold, data_length, n_pos_total=None):
        '''
        :param threshold: margin for squred hinge loss
        '''
        super(MOAPV2LOSS, self).__init__()
        self.u_all = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.u_pos = torch.tensor([0.0]*data_length).view(-1, 1).cuda()
        self.threshold = threshold
        self.n_pos_total = n_pos_total

    def forward(self, f_ps, f_ns, index_s, gamma):
        f_ps = f_ps.view(-1)
        f_ns = f_ns.view(-1)

        vec_dat = torch.cat((f_ps, f_ns), 0)
        mat_data = vec_dat.repeat(len(f_ps), 1)

        f_ps = f_ps.view(-1, 1)

        neg_mask = torch.ones_like(mat_data)
        neg_mask[:, 0:f_ps.size(0)] = 0

        pos_mask = torch.zeros_like(mat_data)
        pos_mask[:, 0:f_ps.size(0)] = 1

        # 3*1 - 3*64 ==> 3*64

        neg_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * neg_mask
        pos_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * pos_mask
        loss = pos_loss + neg_loss

        self.u_pos *= (1 - gamma)
        self.u_all *= (1 - gamma)

        if f_ps.size(0) == 1:
            self.u_pos[index_s] += gamma * self.n_pos_total * (pos_loss.mean()) / len(index_s)
            self.u_all[index_s] += gamma * self.n_pos_total * (loss.mean()) / len(index_s)
        else:
            self.u_all[index_s] += gamma * self.n_pos_total * (loss.mean(1, keepdim=True)) / len(index_s)
            self.u_pos[index_s] += gamma * self.n_pos_total * (pos_loss.mean(1, keepdim=True)) / len(index_s)
        p = (self.u_pos[index_s] - (self.u_all[index_s]) * pos_mask) / (self.u_all[index_s] ** 2)
        p.detach_()
        loss = torch.mean(p * loss)

        return loss

class CrossEntropyLoss(torch.nn.Module):
    """
    Cross Entropy Loss with Sigmoid Function
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = F.binary_cross_entropy_with_logits  # with sigmoid

    def forward(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)


def focal_loss(predScore, target, alpha, gamma, weight):
    """Computes the focal loss"""

    '''
    input_values = -\log(p_t)
    loss = - \alpha_t (1-\p_t)\log(p_t)
    '''
    loss = torch.zeros_like(predScore)
    if len(loss[target == 1])!=0:
        loss[target == 1] = -alpha * (1 - predScore[target == 1]) ** gamma *torch.log(predScore[target == 1]) * weight[target == 1]
    if len(loss[target == 0]) != 0:
        loss[target == 0] = -alpha * (predScore[target == 0]) ** gamma *torch.log(1- predScore[target == 0]) * weight[target == 0]

    return loss.mean()


def get_weight(epoch, beta=0.9999, cls_num_list=None):
    '''

    :param args:
    :param epoch:
    :return: The weights for positive and negative weights
    '''
    if cls_num_list is None:
        cls_num_list = []
    per_cls_weights = None
    if epoch <= 10:
        per_cls_weights = np.ones([2])
    else:
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    return per_cls_weights


class FocalLoss(nn.Module):
    '''
    Input of loss is the output of the model
    '''
    def __init__(self, cls_num_list=None, alpha = 1, gamma=0, reduction ='mean'):
        super(FocalLoss, self).__init__()
        if cls_num_list is None:
            cls_num_list = []
        assert gamma >= 0
        self.gamma = gamma
        self.cls_num_list = cls_num_list
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, input, target, epoch):
        weight = get_weight(epoch, cls_num_list=self.cls_num_list)
        indexTarget=target.detach().cpu().numpy()
        predScore = sigmoidf(input)

        return focal_loss(predScore, target, self.alpha, self.gamma, weight[indexTarget].view(-1,1))
