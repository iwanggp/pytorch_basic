# -*- coding: utf-8 -*-
# @Time    : 2020/3/7 12:19
# @Author  : gpwang
# @File    : criteoptim.py
# @Software: PyCharm
"""
定义优化器和损失函数
"""
import torch.nn as nn
import torch.optim as optim


class Crite_Optim():
    def __init__(self, net):
        self.crite = nn.CrossEntropyLoss()  # 定义损失函数
        self.optimer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, dampening=0.1)  # 选择优化器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimer, step_size=50, gamma=0.1)  # 学习率下降策略
