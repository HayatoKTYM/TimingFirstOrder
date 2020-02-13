import pandas as pd
import numpy as np

from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim

from utils import setup
from train import train
from model import FirstDelayActionPredict_ut_model

def main():
    df_list = setup()

    df = pd.concat(df_list[90:],ignore_index=True)
    x = df.iloc[:,-128:].values
    print(x.shape)
    u = np.clip(1.0 - (df['utter_A'] + df['utter_B']),0,1)
    y = df['action'].map(lambda x:0.8 if x == 'Passive' else 0)

    index = np.where(u==0)[0]
    y[index] = 0.

    y_continue = df['action'].map(lambda x:1 if x == 'Passive-Continue' else 0)
    print('y_continue sum is {}'.format(y_continue.sum()))
    u[np.where(y_continue==1)[0]] = 0
    
    
    df = pd.concat(df_list[90:],ignore_index=True)
    x_val = df.iloc[:,-128:].values
    print(x_val.shape)
    u_val = np.clip(1.0 - (df['utter_A'] + df['utter_B']),0,1)
    y_val = df['action'].map(lambda x:0.8 if x == 'Passive' else 0)

    index = np.where(u_val==0)[0]
    y_val[index] = 0.

    y_continue = df['action'].map(lambda x:1 if x == 'Passive-Continue' else 0)
    print('y_continue sum is {}'.format(y_continue.sum()))
    u_val[np.where(y_continue==1)[0]] = 0
    
    
    net = FirstDelayActionPredict_ut_model()
    print('Model :', net.__class__.__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for name, param in net.named_parameters():
        if 'u_t' in name:
            param.requires_grad = False
            print("勾配計算なし。学習しない：", name)
        elif 'fc' in name or 'lstm' in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False
    
    train_dataloader = data.DataLoader(
        list(zip(x,y)), batch_size=1, shuffle=False)

    test_dataloader = data.DataLoader(
        list(zip(x_val,y_val)), batch_size=1, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": test_dataloader, "test": test_dataloader}
    
    train(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion,optimizer=optimizer,
        num_epochs=2, output='./a_t_model/',resume=True)

if __name__ == '__main__':
    main()