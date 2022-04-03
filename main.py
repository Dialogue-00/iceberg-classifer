import argparse
import torch
import os
import numpy as np
import random
import torchvision
from torchvision import transforms
from torch import optim
from torch.optim import optimizer
from core.dataset import get_dataset
from core.optimizer import get_opt
from model import get_model
import model
from train import train

from utils.seed import setup_seed

# watch -n -1 nvidia-smi

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-m', '--model', type=str, default='vgg16')
    parser.add_argument('-p', '--pretrain', type=bool, default=True)
    parser.add_argument('-e', '--epochs', type=int , default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-d', '--savedir', type=str, default='result/')

    params = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(params)

    save_path = os.path.join(params.model + '-epoch' + str(params.epochs)+ '-' + str(params.batch_size) + '-' + params.optimizer + '-' + str(params.learning_rate) + '-' + 'pre' + '-' + str(params.pretrain))

    if not os.path.exists(params.savedir + save_path):
        os.makedirs(params.savedir + save_path)


    # 引入tensorboard进行可视化
    from torch.utils.tensorboard import SummaryWriter
    log_dir = "runs/" + os.path.join(params.model + '-' + str(params.epochs)+ '-' + str(params.batch_size) + '-' + params.optimizer + '-' + str(params.learning_rate) + '-' + 'pre' + '-' + str(params.pretrain))
    writer = SummaryWriter(log_dir=log_dir)

    # if params.model == 'alexnet':   
    # 图像预处理
    data_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    # 数据集读取
    train_dataloader, val_dataloader = get_dataset(params, data_transformer)

    # 输入图片可视化
    # images, labels = next(iter(train_dataloader))
    # images = images.to(device)
    # grid = torchvision.utils.make_grid(images)
    # writer.add_image('images', grid, 0)

    # 模型读取
    model = get_model(device, params)
    # print(model)                    # 显示模型信息
    # summary(model, (3, 224, 224))   # 显示模型参数

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = get_opt(model, params)
    # lr_scheduler = 

    train(device, model, params, save_path, train_dataloader, val_dataloader, criterion, optimizer, writer)


