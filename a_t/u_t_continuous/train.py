import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
import matplotlib.pyplot as plt
    
def train(net, 
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
    Loss = {'train':[],'val':[]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using',device)
    net.to(device)

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
            threshold = 0.6
            back_cnt = 0
            calc_flag = True 
            for inputs, labels in dataloaders_dict[phase]:    
                inputs = inputs.to(device,dtype=torch.float32)
                labels = labels.to(device,dtype=torch.float32)
                if labels < 0: #会話データの境目に -1 を 挿入:
                    hidden = None
                    continue

                if hidden is None : 
                    out,hidden,a_pre ,u = net(inputs)
                else:
                    out, hidden, a_pre, _ = net(inputs, hidden, out, a_pre) # 順伝播

                if labels >= threshold:
                    l = criterion(out, labels) # ロスの計算
                    loss += l
                    epoch_loss += l.item() * inputs.size(0)  # lossの合計を更新
                    train_cnt += 1
                    back_cnt += 1
                
                if out < threshold: # 計算フラグ初期化
                    calc_flag = True 
                
                elif out >= threshold and calc_flag: # 1回計算したらそのあとのyt > threshold は計算しない
                    l = criterion(out, out.data - 0.2) # ロスの計算
                    loss += l
                    epoch_loss += l.item() * inputs.size(0)  # lossの合計を更新
                    train_cnt += 1
                    back_cnt += 1
                
                if phase == 'train' and back_cnt ==1:#(labels >= threshold or out>=threshold): # 訓練時はバックプロパゲーション 
                    optimizer.zero_grad() # 勾配の初期化
                    loss.backward() # 勾配の計算
                    optimizer.step()# パラメータの更新
                    loss = 0 #累積誤差reset
                    back_cnt = 0
                    
                    out = out.detach()
                    hidden = (hidden[0].detach(),hidden[1].detach())
                    a_pre = a_pre.detach()
                    
            epoch_loss = epoch_loss /  train_cnt
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            Loss[phase].append(epoch_loss)
            
        if resume:
            torch.save(net.state_dict(), output+'epoch_' + str(epoch+1) + '_ut_train.pth')
            hidden = None
            y_true, y_pred = np.array([]), np.array([])
            all_inputs = []
            a_values_list = np.array([])
            u_values_list = np.array([])
            for inputs,labels in dataloaders_dict['test']:          
                inputs = inputs.to(device,dtype=torch.float32)
                if labels < 0: #会話データの境目に -1 を 挿入:
                    hidden = None
                    continue
                if hidden is None : 
                    out,hidden,a ,u= net(inputs)
                else:
                    out,hidden,a,u = net(inputs,hidden,out,a)       

                y_true = np.append(y_true, labels.data.numpy())
                y_pred = np.append(y_pred, out.cpu().data.numpy())
                a_values_list = np.append(a_values_list, a.cpu().data.numpy())
                u_values_list = np.append(u_values_list, u.cpu().data.numpy())

        
            plt.figure(figsize=(15,4))
            plt.plot(y_true[:300],label = 'true label')
            plt.plot(y_pred[:300],label = 'predict')
            plt.plot(u_values_list[:300],label='u_t',color='g')
            plt.plot(a_values_list[:300],label='a_t',color='r')
            plt.legend()
            plt.savefig(os.path.join(output,'result_{}.png'.format(epoch+1)))
        
        print('-------------')
     
    if resume: # 学習過程(Loss) を保存するか    
        plt.figure(figsize=(15,4))
        plt.plot(Loss['val'],label='val')
        plt.plot(Loss['train'],label='train')
        plt.legend()
        plt.savefig(os.path.join(output,'history.png'))
    
    
    y_true, y_pred = np.array([]), np.array([])
    y_prob = np.array([])
    all_inputs = []

    hidden = None
    y_true, y_pred = np.array([]), np.array([])
    all_inputs = []
    a_values_list = np.array([])
    u_values_list = np.array([])

    hidden = None
    for inputs, labels in dataloaders_dict['test']:
        inputs = inputs.to(device,dtype=torch.float32)
        if labels < 0: #会話データの境目に -1 を 挿入:
            hidden = None
            continue
        if hidden is None : 
            out,hidden,a ,u= net(inputs)
        else:
            out,hidden,a,u = net(inputs,hidden,out,a)     

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, out.cpu().data.numpy())
        a_values_list = np.append(a_values_list, a.cpu().data.numpy())
        u_values_list = np.append(u_values_list, u.cpu().data.numpy())

    if resume: # 出力の可視化例を保存するかどうか
        plt.figure(figsize=(15,4))
        plt.plot(y_true[:300],label = 'true label')
        plt.plot(y_pred[:300],label = 'predict')
        plt.plot(u_values_list[:300],label='u_t',color='g')
        plt.plot(a_values_list[:300],label='a_t',color='r')
        plt.legend()
        plt.savefig(os.path.join(output,'result.png'))