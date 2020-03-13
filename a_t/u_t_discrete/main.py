from model import FirstDelayActionPredict_VAD
from train import train_first_delay
from utils import setup

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

def hang_over(y):
    """
    u の末端 200 ms を １ にする
    """
    print('before',y.sum())
    for i in range(len(y)-1):
        if y[i] == 0 and y[i+1] == 1:
            y[i-2:i+1] = 1.
    print('after',y.sum())
    return y

def u_t_maxcut(u,max_frame=20):
    count = 0
    for i in range(len(u)):
        if u[i] != 1:
            count = 0
        else:
            count += 1
            if count > max_frame:
                u[i] = 0
        return u

def main():
    df_list = setup() 
    
    #train
    train_id = 89
    df = pd.concat(df_list[:train_id],ignore_index=True)
    x = df.iloc[:,-128:].values
    
    print(x.shape)
    u = np.clip(1.0 - (df['utter_A'] + df['utter_B'] ),0,1)
    u = u_t_maxcut(u,max_frame=30)
    # 行動開始 -> 0.8  会話の境目 -> -1 それ以外 0
    y = df['action'].map(lambda x:0.8 if x == 'Passive' else 0.8 if x == 'Active' else -1 if x == -1 else 0)
    # u(t) = 0 なら 教師も 0 にしとく (後々はちゃんと学習でなんとかしたい)
    index = np.where(u==0)[0]
    y[index] = 0.
    print(y.sum())

    # sys発話中は u(t) = 0 にする
    
    

    flag = [False] + [True if (u[i] == 1 and u[i+1] == 1) else False for i in range(len(u)-1)]
    # val/ test
    df = pd.concat(df_list[train_id:],ignore_index=True)
    x_val = df.iloc[:,-128:].values
    print(x_val.shape)
    u_val = np.clip(1.0 - (df['utter_A'] + df['utter_B'] ),0,1)
    print(u_val.sum())
    u_val = u_t_maxcut(u_val,max_frame=30)
    print(u_val.sum())

    # 行動開始 -> 0.8  会話の境目 -> -1 それ以外 0
    y_val = df['action'].map(lambda x:0.8 if x == 'Passive' else 0.8 if x == 'Active' else -1 if x == -1 else 0)
    # u(t) = 0 なら 教師も 0 にしとく (後々はちゃんと学習でなんとかしたい)
    index = np.where(u_val==0)[0]
    y_val[index] = 0.

    # sys発話中は u(t) = 0 にする
    y_continue = df['action'].map(lambda x:1 if x == 'Passive-Continue' else 1 if x == 'Active-Continue'else 0)
    print('y_continue sum is {}'.format(y_continue.sum()))
    u_val[np.where(y_continue==1)[0]] = 0

    flag_val = [False] + [True if u_val[i] == 1 and u_val[i+1] == 1 else False for i in range(len(u_val)-1) ]
    
    net = FirstDelayActionPredict_VAD(input_size=128,hidden_size=128)
    print('Model :', net.__class__.__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-03)
    optimizer = optim.SGD(net.parameters(), lr=1e-04)
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
        num_epochs=100, output='./a_t_model_thres0.8_reset_all_bpcnt_32_one_calc_ver_ut_max_frame30_SGD/', 
        resume=True)
if __name__ == '__main__':
    main()
