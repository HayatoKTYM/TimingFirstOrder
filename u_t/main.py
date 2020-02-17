from model import TimeActionPredict
from train import train
from utils import setup

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np


def main():
    """
    x .. 入力特徴量
    y .. 教師ラベル(0/1)
    LSTM に通さず，フレーム毎に独立に推定するので，バッチサイズは1じゃなくて良い
    """
    df_list = setup()

    df = pd.concat(df_list[:90])
    x = df.iloc[:,-128:].values
    
    print(x.shape)
    y = np.clip(1.0 - (df['utter_A'] + df['utter_B']),0,1)
    
    
    df = pd.concat(df_list[90:])
    x_val = df.iloc[:,-128:].values
    print(x_val.shape)
    y_val = np.clip(1.0 - (df['utter_A'] + df['utter_B']),0,1)
    
    net = TimeActionPredict()
    print('Model :', net.__class__.__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0]).to(device))
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for name, param in net.named_parameters():
        if 'fc' in name or 'lstm' in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False
        
    train_dataloader = data.DataLoader(
        list(zip(x,y)), batch_size=1, shuffle=False)

    test_dataloader = data.DataLoader(
        list(zip(x_val,y_val)), batch_size=1, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": test_dataloader, "test": test_dataloader}
    
    train(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion,optimizer=optimizer,
        num_epochs=50, output='./lstm_model/', resume=True)

if __name__ == '__main__':
    main()
