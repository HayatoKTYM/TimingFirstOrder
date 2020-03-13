import pandas as pd
import numpy as np

from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim

from utils import setup
from train import train
from model import FirstDelayActionPredict_ut_model

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

def main():
    df_list = setup()
    train_id = 89
    df = pd.concat(df_list[:train_id],ignore_index=True)
    x = df.iloc[:,-128:].values
    print(x.shape)
    u = hang_over(np.clip(1.0 - (df['utter_A'] + df['utter_B'] ),0,1))
    # 行動開始 -> 0.6  会話の境目 -> -1 それ以外 0
    y = df['action'].map(lambda x:0.6 if x == 'Passive' else 0.6 if x == 'Active' else -1 if x == -1 else 0)
    index = np.where(u==0)[0]
    y[index] = 0.

    y_continue = df['action'].map(lambda x:1 if x == 'Passive-Continue' else 0)
    print('y_continue sum is {}'.format(y_continue.sum()))
    u[np.where(y_continue==1)[0]] = 0
    
    
    df = pd.concat(df_list[train_id:],ignore_index=True)
    x_val = df.iloc[:,-128:].values
    print(x_val.shape)
    u_val = hang_over(np.clip(1.0 - (df['utter_A'] + df['utter_B'] ),0,1))
    # 行動開始 -> 0.6  会話の境目 -> -1 それ以外 0
    y_val = df['action'].map(lambda x:0.6 if x == 'Passive' else 0.6 if x == 'Active' else -1 if x == -1 else 0)
    index = np.where(u_val==0)[0]
    y_val[index] = 0.

    y_continue = df['action'].map(lambda x:1 if x == 'Passive-Continue' else 0)
    print('y_continue sum is {}'.format(y_continue.sum()))
    u_val[np.where(y_continue==1)[0]] = 0
    
    net = FirstDelayActionPredict_ut_model(input_size=128,PATH='../../u_t_each/lstm_model_all_reset/epoch_29_ut_train.pth',lstm_model=True)
    print('Model :', net.__class__.__name__)

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
    
    assert len(x) == len(y); print('problem occurred!! please check your dataset length..')
    train_dataloader = data.DataLoader(
        list(zip(x,y)), batch_size=1, shuffle=False)

    test_dataloader = data.DataLoader(
        list(zip(x_val,y_val)), batch_size=1, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": test_dataloader, "test": test_dataloader}
    
    train(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion,optimizer=optimizer,
        num_epochs=100, output='./a_t_model_bpcnt_1_thres0.6_ut-notlessthan0.5_ut_lstm_model/',resume=True)

if __name__ == '__main__':
    main()