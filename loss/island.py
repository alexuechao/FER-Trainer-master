#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
岛屿损失旨在减少类内变化，同时扩大类间差异
目的是在center loss的基础上, 进一步优化类间距离
https://blog.csdn.net/heruili/article/details/88912074
Loss = L_softmax + lamda * L_island
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class IslandLoss(nn.Module):
    def __init__(self, features_dim=2, num_class=7, alpha1=0.01, scale=1.0, batch_size=64):
        """
        初始化
        :param features_dim: 特征维度 = c*h*w
        :param num_class: 类别数量
        :param alpha:   island loss的权重系数 [0,1]
        """
        assert 0 <= alpha1 <= 1
        super(IslandLoss, self).__init__()
        self.alpha1 = alpha1
        self.num_class = num_class
        self.scale = scale
        self.batch_size = batch_size
        self.feat_dim = features_dim
        # store the center of each class , should be ( num_class, features_dim)
        #self.feature_centers = nn.Parameter(torch.randn([num_class, features_dim]))
        self.feature_centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).cuda())

        # self.lossfunc = CenterLossFunc.apply
        #init_weight(self, 'normal')

    def forward(self, output_features, y_truth):
        """
        损失计算
        :param output_features: conv层输出的特征,  [b,c,h,w]
        :param y_truth:  标签值  [b,]
        :return:
        """
        batch_size = y_truth.size(0)
        output_features = output_features.view(batch_size, -1)
        assert output_features.size(-1) == self.feat_dim

        centers_pred = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
        diff = output_features - centers_pred
        # 1 先求 center loss
        loss_center = 1 / 2.0 * (diff.pow(2).sum()) / self.batch_size

        # 2 再求 类心余弦距离
        # 每个类心求余弦距离，+1 使得范围为0-2，越接近0表示类别差异越大，从而优化Loss即使得类间距离变大。
        centers = self.feature_centers
        # Ci X Ci.T
        centers_mm = centers.mm(centers.t())  # [num_class, num_class]

        # 求出每个类别的向量模长 ||Ci||
        centers_mod = torch.sum(centers * centers, dim=1, keepdim=True).sqrt()  # [num_class, 1]
        centers_mod_mm = centers_mod.mm(centers_mod.t())  # [num_class,num_class]

        # 求出 cos距离 矩阵, 这是一个对称矩阵
        centers_cos_dis = centers_mm / centers_mod_mm

        # 将对角线上元素置0, 代表同一个类别的距离不考虑
        angle_mtx = torch.eye(self.num_class)  # 对角线为1,
        mask = ~angle_mtx.gt(0)
        mask = angle_mtx.masked_fill_(mask, 1) * mask  # 对角线为0, 其他为1

        centers_cos_dis += 1
        centers_cos_dis *= mask.cuda()

        sum_centers_cos_dis = centers_cos_dis.sum() / 2
        loss_island = loss_center + self.alpha1 * sum_centers_cos_dis
        return loss_island


if __name__ == '__main__':
    import random

    # test 1
    num_class = 10
    batch_size = 10
    feat_dim = 5
    ct = IslandLoss(feat_dim, num_class, 0.1, 1., batch_size)
    y = torch.Tensor([random.choice(range(num_class)) for i in range(batch_size)])
    feat = torch.zeros(num_class, feat_dim).requires_grad_()
    print(list(ct.parameters()))
    print(ct.feature_centers.grad)
    out = ct(feat, y)
    print(out.item())
    out.backward()
    print(ct.feature_centers.grad)
    print(feat.grad)