# -*- coding: utf-8 -*-
# @Time    : 2020/3/7 01:29
# @Author  : gpwang
# @File    : train.py
# @Software: PyCharm

import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from crite_optim.criteoptim import Crite_Optim

sys.path.append(".")
from modeling.cifar10_model import Net
from mydatasets.MyDatasets import MyDatasets

train_txt_path = "./data/cifar-10-png/train.txt"
valid_txt_path = "./data/cifar-10-png/valid.txt"
# 配置参数
train_batch_size = 16
valid_batch_size = 16
lr_init = 0.001
max_epoch = 2

# ----------------------------------step 1/5:加载数据------------------------------
# 数据预处理设置
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])
validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])
# 构建MyDataset实例
train_data = MyDatasets(train_txt_path, transform=trainTransform)
valid_data = MyDatasets(valid_txt_path, transform=validTransform)
# 构建DataLoader
train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=valid_batch_size)
# ----------------------------------step 2/5:定义模型------------------------------
net = Net()  # 实例化一个网络

net.initialize_weights()
# ----------------------------------step 3/5:定义损失函数和优化器------------------------------------
# criterion = nn.CrossEntropyLoss()  # 选择损失函数
# optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)  # 选择优化器
cri_optim = Crite_Optim(net)
criterion = cri_optim.crite
optimizer = cri_optim.optimer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略

# ----------------------------------step 4/5:训练------------------------------------
for epoch in range(max_epoch):
    # print("Epoch {}/{}".format(epoch + 1, max_epoch))
    # print("-" * 10)
    loss_sigma = 0.0  # 记录一个epoch的loss之和
    correct = 0.0  # 正确的个数
    total = 0.0  # 总共正确的个数
    # ----------------------改进
    running_loss = 0.0
    running_corrects = 0

    # -----------------------
    # 在训练集训练数据
    for i, data in enumerate(train_loader):
        inputs, labels = data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.train()
        # inputs, labels = Variable(inputs), Variable(labels)#目前已经被丢弃了，以后不用这种方法的。
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播，后向传播，更新权重
        optimizer.zero_grad()  # 已经要将优化器置0，防止后面累加

        # ---------------------------------------------------------改进
        with torch.set_grad_enabled(True):
            outputs = net(inputs)  # 从模型中获得输出
            loss = criterion(outputs, labels)  # 获得损失值
            _, preds = torch.max(outputs, 1)
            loss.backward()  # 反向传播
            optimizer.step()  # 学习率更新
            scheduler.step()  # 更新学习率，在PyTorch1.0以后学习率策略的更新必须放到学习率更新的后面。这是官方建议这样做的
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)
        total += labels.size(0)
        if i % 10 == 9:
            loss_avg = running_loss / 10
            running_loss = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, running_corrects / total))
    # 每一个Epoch打印验证集上的损失
    loss_sigma = 0.0
    valid_correct = 0
    for i, data in enumerate(valid_loader):
        net.eval()
        # 获取图片和标签
        images, labels = data
        # forward
        outputs = net(images)
        # 计算loss
        loss = criterion(outputs, labels)
        loss_sigma += loss.item() * images.size(0)
        # 统计
        _, predicted = torch.max(outputs.data, 1)
        valid_correct += torch.sum(labels == predicted)
        valid_loss = loss_sigma / len(valid_loader.dataset)
        valid_acc = valid_correct.double() / len(valid_loader.dataset)
    print("Valid set Loss is {:.4f} Accuracy:{:.2f}".format(valid_loss, valid_acc))
print("Finished Training......")
net_save_path = 'model.pth'
torch.save(net.state_dict(), net_save_path)
