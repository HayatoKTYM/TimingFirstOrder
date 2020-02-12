"""
学習ファイル
train.py batch_size = 1, frames = 1 で LSTM に入力するプログラム
BP は 64回 feed farward するごとに行う (train_cnt % 64 == 0)
"""
import torch
import torch.nn as nn

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
def train(net, 
        dataloaders_dict, 
        criterion, optimizer,
        num_epochs=10,
        output='./',
        resume=True,         
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

    Acc = {'train': [0]*num_epochs,
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
            epoch_corrects = 0  # epochの正解数
            hidden = None # lstmの初期状態
            loss = 0 # 損失和
            train_cnt = 0 # feed farward 回数

            for inputs, labels in dataloaders_dict[phase]:    
                inputs = inputs.to(device,dtype=torch.float32)
                labels = labels.to(device,dtype=torch.long)
                if hidden is None:
                    out,hidden= net(inputs,None) # 順伝播
                else:
                    out,hidden = net(inputs,hidden) # 順伝播
                train_cnt += 1
                l = criterion(out, labels) # ロスの計算
                loss += l
                _, preds = torch.max(out, 1)  # ラベルを予測
                
                if phase == 'train' and train_cnt % 64 == 0 : # 訓練時はバックプロパゲーション
                    optimizer.zero_grad() # 勾配の初期化
                    loss.backward() # 勾配の計算
                    optimizer.step()# パラメータの更新
                    loss = 0 #累積誤差reset
                    
                    hidden = (hidden[0].detach(),hidden[1].detach()) # BPのつながりをカット

                epoch_loss += l.item() * inputs.size(0)  # lossの合計を更新
                epoch_corrects += torch.sum(preds == labels.data) # 正解数の合計を更新
                
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            Loss[phase][epoch] = epoch_loss 
            Acc[phase][epoch] = float(epoch_acc.cpu().numpy())
            
        if resume:
            torch.save(net.state_dict(), os.path.join(output,'epoch_' + str(epoch+1) + '_ut_train.pth'))
        
        y_true, y_pred = np.array([]), np.array([])
        # precision , recall , F1-score, confusion_matrix を表示
        hidden = None
        for inputs, labels in dataloaders_dict['test']:
            inputs = inputs.to(device,dtype=torch.float32)
            if hidden is None:
                out,hidden = net(inputs)
            else:
                out,hidden = net(inputs,hidden)
            _, preds = torch.max(out, 1)  # ラベルを予測
            y_true = np.append(y_true, labels.data.numpy())
            y_pred = np.append(y_pred, preds.cpu().data.numpy())
        
        print(accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))
        print('-------------')

    if resume: # 学習過程(Loss) を保存するか    
        plt.figure(figsize=(15,4))
        plt.plot(Loss['val'],label='val')
        plt.plot(Loss['train'],label='train')
        plt.legend()
        plt.savefig(os.path.join(output,'history.png'))
    
    print('training finish and save train history...')
    y_true, y_pred = np.array([]), np.array([])
    y_prob = np.array([])
    
    hidden = None
    for inputs, labels in dataloaders_dict['test']: # 評価
        inputs = inputs.to(device,dtype=torch.float32)
        if hidden is None:
            out,hidden = net(inputs)
        else:
            out,hidden = net(inputs,hidden)
        _, preds = torch.max(out, 1)  # ラベルを予測

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())
        y_prob = np.append(y_prob, nn.functional.softmax(out).cpu().data.numpy()[0][1])

    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    
    if resume: # 出力の可視化例を保存するかどうか
        plt.figure(figsize=(15,4))
        plt.plot(y_true[:300],label = 'true label')
        plt.plot(y_prob[:300],label = 'predict')
        plt.legend()
        plt.savefig(os.path.join(output,'result.png'))
