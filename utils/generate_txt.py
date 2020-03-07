# -*- coding: utf-8 -*-
# @Time    : 2020/3/6 23:43
# @Author  : gpwang
# @File    : generate_txt.py
# @Software: PyCharm
"""
训练文本文件，大多数加载训练文件都是以文本的形式
我们生成训练文本的格式是下面这个样子的
../data/cifar-10-png/raw_train/**.png label
"""
import glob
import os

train_path = '../data/cifar-10-png/train'
train_txt = '../data/cifar-10-png/train.txt'
valid_path = '../data/cifar-10-png/valid'
valid_txt = '../data/cifar-10-png/valid.txt'
test_path = '../data/cifar-10-png/test'
test_txt = '../data/cifar-10-png/test.txt'


def generate_txt(txt_path, data_path):
    for root, dirs, _ in os.walk(data_path, topdown=True):
        for dir in dirs:
            img_list = glob.glob(os.path.join(root, dir, "*.png"))
            lable_name = str(dir)
            for img in img_list:
                if not img.endswith("png"):
                    continue
                line = img + " " + lable_name + "\n"
                with open(txt_path, 'a+') as f:
                    f.write(line)


if __name__ == '__main__':
    generate_txt(train_txt, train_path)
    generate_txt(valid_txt, valid_path)
    generate_txt(test_txt, test_path)
