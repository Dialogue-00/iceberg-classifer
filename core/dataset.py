from numpy.core.fromnumeric import size
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import pandas as pd 
import math
import argparse
import os

from utils.seed import setup_seed

class IceShipDataset(Dataset):
    def __init__(self, datapath, transforms=None):
        self.datapath = datapath
        self.transforms = transforms
        self.data = pd.read_json(self.datapath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 获取一张图片
        b_1 = np.array(self.data.iloc[index]['band_1']).reshape(75, 75).astype(np.float32)
        b_2 = np.array(self.data.iloc[index]['band_2']).reshape(75, 75).astype(np.float32)
        b_3 = (b_1 + b_2) / 2.0
        r = (b_1 + abs(b_1.min())) / np.max((b_1 + abs(b_1.min())))
        g = (b_2 + abs(b_2.min())) / np.max((b_2 + abs(b_2.min()))) 
        b = (b_3 + abs(b_3.min())) / np.max((b_3 + abs(b_3.min())))
        full_img = np.stack([r, g, b], axis=2)
        full_img*=255.0
        # full_img*=250.0
        full_img = full_img.astype(np.uint8)
        # full_img = Image.fromarray(full_img.astype(np.uint8))
        # full_img = Image.fromarray(full_img)
        # full_img = full_img.convert('RGB')
        if self.transforms is not None:
            img_as_tensor = self.transforms(full_img)
        # 获取一个标签
        img_label = self.data['is_iceberg'].values[index]
        return (img_as_tensor, img_label)



def get_dataset(params, transfomer):
    datapath = './data/train/data/processed/train.json'
    dataset = IceShipDataset(datapath, transfomer)
    len_dataset = len(dataset)
    setup_seed(params)
    # 划分比例可更改
    len_train = math.ceil(0.8 * len_dataset)
    len_val = len_dataset - len_train
    # 划分训练集和验证集
    train_ds, val_ds = random_split(dataset, [len_train, len_val])

    nw = min([os.cpu_count(), params.batch_size if params.batch_size > 1 else 0, 8])
    train_dl = DataLoader(train_ds,batch_size=params.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=params.batch_size, shuffle=False)

    # train_dl = DataLoader(
    # train_ds,
    # batch_size=params.batch_size,
    # shuffle=True,
    # pin_memory=True,
    # num_workers=nw,
    # )

    return train_dl, val_dl


if __name__ == '__main__':
    data_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    params = parser.parse_args()
    train_dataloader, val_dataloader = get_dataset(params, data_transformer)
    print(params.batch_size)
    print(len(train_dataloader.dataset))
