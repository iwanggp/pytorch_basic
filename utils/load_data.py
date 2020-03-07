# -*- coding: utf-8 -*-
# @Time    : 2020/3/6 23:12
# @Author  : gpwang
# @File    : load_data.py
# @Software: PyCharm
"""
从压缩文件中加载数据
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append("..")  # 添加根目录
dataset_dir = '../data/cifar-10-batches-py/test_batch'
train_set_dir = os.path.join("..", "data", "cifar-10-png", "raw_train")
test_set_dir = os.path.join("..", "data", "cifar-10-png", "raw_test")


# 解压文件为python可以读取
def unpicle(file):
    with open(file, 'rb') as f:
        _dict = pickle.load(f, encoding='bytes')
    return _dict


# 创建文件夹的函数
def mkdirs(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


if __name__ == '__main__':
    dataset = unpicle(dataset_dir)
    for i in tqdm(range(10000)):
        img = np.reshape(dataset[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        label = str(dataset[b'labels'][i])
        o_dir = os.path.join(train_set_dir, label)
        mkdirs(o_dir)
        img_name = label + "_" + str(i) + ".png"
        img_path = os.path.join(o_dir, img_name)
        plt.imsave(img_path, img)
