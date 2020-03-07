# -*- coding: utf-8 -*-
# @Time    : 2020/3/7 00:49
# @Author  : gpwang
# @File    : MyDatasets.py
# @Software: PyCharm
"""
构建自己的数据集类，需要实现Datasets这个类。需要关键实现__getitem__和__len__方法
1.制作图片数据的索引
2.构建Dataset子类
"""
from PIL import Image
from torch.utils.data import Dataset


class MyDatasets(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        imgs = []
        with open(txt_path, 'r') as f:
            fh = f.readlines()
            for line in fh:
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform  # tansform图片变换
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert("RGB")  # 这里的图片是一个PIL对象
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
