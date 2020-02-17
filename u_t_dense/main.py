from model import U_t_train
from train import train
from utils import setup

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

from torch.utils import data

def main():
    """
    x .. 入力特徴量
    y .. 教師ラベル(0 or 1)
    stateful(状態を常に保持)なので，バッチサイズは1としている
    """
    df_list = setup()
    
    df = pd.concat(df_list[:90])
    x = df.iloc[:,-128:-64].values
    print(x.shape)
    x = np.append(x,df.iloc[:,-64:].values,axis=0)
    print(x.shape)
    y = 1.0 - df['utter_A']
    y = np.append(y,1.0 - df['utter_B'])
    
    df = pd.concat(df_list[90:])
    x_val = df.iloc[:,-128:-64].values
    x_val = np.append(x_val,df.iloc[:,-64:].values,axis=0)
    print(x_val.shape)
    y_val = 1.0 - df['utter_A']
    y_val = np.append(y_val,1.0 - df['utter_B'])
    
    net = U_t_train()
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
        list(zip(x,y)), batch_size=128, shuffle=True)

    test_dataloader = data.DataLoader(
        list(zip(x_val,y_val)), batch_size=128, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": test_dataloader, "test": test_dataloader}
    
    train(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion,optimizer=optimizer,
        num_epochs=1, output='./dense_model/', resume=True)

if __name__ == '__main__':
    main()
