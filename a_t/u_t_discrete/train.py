import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import glob
import os
import matplotlib
import matplotlib.pyplot as plt

def train_first_delay(net, 
                    dataloaders_dict, 
                    criterion, optimizer,
                    num_epochs=10,
                    output='./',
                    resume=False, 
    ):
    """
    学習ループ
    """
    os.makedirs(output,exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using',device)
    net.to(device)

    Loss = {'train': [0]*num_epochs,
            'val': [0]*num_epochs}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            
            hidden = None
            loss = 0
            train_cnt = 0
            out_cnt = 0
            threshold = 0.8
            for inputs,u , labels,flag in dataloaders_dict[phase]:    
                
                inputs = inputs.to(device,dtype=torch.float32)
                u = u.to(device,dtype=torch.float32)
                labels = labels.to(device,dtype=torch.float32)
                if hidden is None:
                    out,hidden,a = net(inputs,u) # 順伝播
                else:                
                    out,hidden,a = net(inputs,u,hidden,out,a) # 順伝播

                if labels >= 0.8 :
                    loss = criterion(out, labels) # ロスの計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    train_cnt += 1
                    #print('out {}'.format(out))
                elif out >= threshold:
                    loss = criterion(out, out.data-0.2) # ロスの計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    train_cnt += 1
                    out_cnt += 1
                    
                if phase == 'train' and (labels>=0.8 or out >= threshold) : # 訓練時はバックプロパゲーション 
                    #print('loss {}, out {}'.format(loss,out))
                    optimizer.zero_grad() # 勾配の初期化
                    loss.backward() # 勾配の計算
                    optimizer.step()# パラメータの更新
                    loss = 0 #累積誤差reset
                    
                    out = out.detach()
                    a = a.detach()
                    hidden = (hidden[0].detach(),hidden[1].detach())
                    
                    
            epoch_loss = epoch_loss /  train_cnt
            print('{} Loss: {:.4f} out_cnt:{}'.format(phase, epoch_loss,out_cnt))
            Loss[phase][epoch] = epoch_loss 
            
        if resume:
            torch.save(net.state_dict(), output+'epoch_' + str(epoch+1) + '_ut_train.pth')
        print('-------------')
    
    if resume: # 学習過程(Loss) を保存するか    
        plt.figure(figsize=(15,4))
        plt.plot(Loss['val'],label='val')
        plt.plot(Loss['train'],label='train')
        plt.legend()
        plt.savefig(os.path.join(output,'history.png'))
        
    y_true, y_pred = np.array([]), np.array([])
    all_inputs = []
    a_values_list = np.array([])

    hidden = None
    for inputs, u, labels,flag in dataloaders_dict['test']:
        inputs = inputs.to(device,dtype=torch.float32)
        u = u.to(device,dtype=torch.float32)
        if hidden is None:
            out,hidden,a = net(inputs,u)
        else:
            out,hidden,a = net(inputs,u,hidden,out.detach(),a)     

        for x in u.cpu().numpy():
            all_inputs.append(x*0.4)
        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, out.cpu().data.numpy())
        a_values_list = np.append(a_values_list, a.cpu().data.numpy())

    if resume: # 出力の可視化例を保存するかどうか
        plt.figure(figsize=(15,4))
        plt.plot(y_true[:300],label = 'true label')
        plt.plot(y_pred[:300],label = 'predict')
        plt.plot(all_inputs[:300],label='u_t',color='g')
        plt.plot(a_values_list[:300],label='a_t',color='r')
        plt.legend()
        plt.savefig(os.path.join(output,'result.png'))
