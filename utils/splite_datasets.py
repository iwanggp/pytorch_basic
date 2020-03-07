# -*- coding: utf-8 -*-
# @Time    : 2020/3/7 00:02
# @Author  : gpwang
# @File    : splite_datasets.py
# @Software: PyCharm
"""
数据解压出来后，下一步就要划分数据集。有很多划分数据集的方法，这里我们也自己实现该方法
"""
import glob
import os
import random
import shutil

dataset_dir = '../data/cifar-10-png/raw_train'
train_dir = '../data/cifar-10-png/train'
valid_dir = '../data/cifar-10-png/valid'
test_dir = '../data/cifar-10-png/test'
# 这里按照8：1：1的比例进行划分的
train_per = 0.8
valid_per = 0.1
test_per = 0.1


# 创建文件夹的函数
def mkdirs(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


def split_dataset(dataset_dir):
    for root, dirs, _ in os.walk(dataset_dir):
        for dir in dirs:
            img_list = glob.glob(os.path.join(root, dir, "*.png"))
            random.seed(666)
            random.shuffle(img_list)
            img_num = len(img_list)
            train_point = int(img_num * train_per)
            valid_point = int(img_num * (train_per + valid_per))
            for i in range(img_num):
                if i < train_point:
                    out_dir = os.path.join(train_dir, dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, dir)
                else:
                    out_dir = os.path.join(test_dir, dir)
                mkdirs(out_dir)
                out_path = os.path.join(out_dir, os.path.split(img_list[i])[-1])
                shutil.copy(img_list[i], out_path)
            print('Class:{}, train:{}, valid:{}, test:{}'.format(dir, train_point, valid_point - train_point,
                                                                 img_num - valid_point))


if __name__ == '__main__':
    split_dataset(dataset_dir)
