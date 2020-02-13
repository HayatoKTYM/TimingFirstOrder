from model import FirstDelayActionPredict_VAD
from train import train_first_delay
from utils import setup

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np


def main():
    df_list = setup() 
    
    #train
    train_id = 90
    df = pd.concat(df_list[train_id:],ignore_index=True)
    x = df.iloc[:,-128:].values
    
    print(x.shape)
    u = np.clip(1.0 - (df['utter_A'] + df['utter_B'] ),0,1)
    y = df['action'].map(lambda x:0.8 if x == 'Passive' else 0)
    # u(t) = 0 なら 教師も 0 にしとく (後々はちゃんと学習でなんとかしたい)
    index = np.where(u==0)[0]
    y[index] = 0.
    print(y.sum())

    # sys発話中は u(t) = 0 にする
    y_continue = df['action'].map(lambda x:1 if x == 'Passive-Continue' else 0)
    print('y_continue sum is {}'.format(y_continue.sum()))
    u[np.where(y_continue==1)[0]] = 0

    flag = [False] + [True if (u[i] == 1 and u[i+1] == 1) else False for i in range(len(u)-1)]
    # val/ test
    df = pd.concat(df_list[train_id:],ignore_index=True)
    x_val = df.iloc[:,-128:].values
    print(x_val.shape)
    u_val = np.clip(1.0 - (df['utter_A'] + df['utter_B'] ),0,1)
    y_val = df['action'].map(lambda x:0.8 if x == 'Passive' else 0)
    # u(t) = 0 なら 教師も 0 にしとく (後々はちゃんと学習でなんとかしたい)
    index = np.where(u_val==0)[0]
    y_val[index] = 0.
    # sys発話中は u(t) = 0 にする
    y_continue = df['action'].map(lambda x:1 if x == 'Passive-Continue' else 0)
    print('y_continue sum is {}'.format(y_continue.sum()))
    u_val[np.where(y_continue==1)[0]] = 0

    flag_val = [False] + [True if u_val[i] == 1 and u_val[i+1] == 1 else False for i in range(len(u_val)-1) ]
    
    net = FirstDelayActionPredict_VAD()
    print('Model :', net.__class__.__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for name, param in net.named_parameters():
        if 'fc' in name or 'lstm' in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False
    
    train_dataloader = data.DataLoader(
        list(zip(x,u,y,flag)), batch_size=1, shuffle=False)

    test_dataloader = data.DataLoader(
        list(zip(x_val,u_val,y_val,flag_val)), batch_size=1, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": test_dataloader, "test": test_dataloader}
    
    train_first_delay(net=net, dataloaders_dict=dataloaders_dict, 
        criterion=criterion,optimizer=optimizer,
        num_epochs=2, output='./a_t_model/', 
        resume=True)
if __name__ == '__main__':
    main()