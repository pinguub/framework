"""main.py文件是主要执行文件，在这里调用工具部署工作"""
import torch # 训练框架
import torch.optim as optim # 优化器
from torch.utils.data import DataLoader # 数据加载器，也可以自己写
from torch.utils.tensorboard import SummaryWriter # 训练日志及可视化

import tqdm # 进度条库
import numpy as np
import os
from utils import *

def train_model(model, optimizer, train_loader, test_loader, step, num_epochs,state_save_path):
    """定义训练方法"""
    model.train(train_loader)

def val_model(model, val_loader):
    """定义测试方法"""
    model.eval()

if __name__ == '__main__':

