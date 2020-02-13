import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstDelayActionPredict_VAD(nn.Module):
    """
    行動予測するネットワーク
    時定数 a(t) を　学習する
    u(t) .. 非発話らしさ (step応答)
    a(t),u(t) から 1次遅れ系のstep応答を計算し， 行動生成らしさ y(t) を計算
    u(t) は 0 / 1 の離散値を入力する version
    """
    def __init__(self, num_layers = 1, input_size=128, hidden_size = 64):
        super(FirstDelayActionPredict_VAD, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()
        
        self.lstm = torch.nn.LSTM(
            input_size = hidden_size, #入力size
            hidden_size = hidden_size, #出力size
            batch_first = True, # given_data.shape = (batch , frames , input_size)
        )
        
        #self.fc2 = nn.Linear(hidden_size, 2)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.count = 0
        #self.a = 0

    def forward(self, x, u, hidden=None, y_pre=0, a_pre=0):
        assert len(x.shape) == 2 , print('data shape is incorrect.')
        x = self.dr1(self.relu1(self.fc1(x)))
        x = x.view(1, 1, -1)
        if hidden is None:
            hidden = self.reset_state()
            print('reset state!!')

        h, hidden = self.lstm(x, hidden)
        a = F.sigmoid(self.fc3(h[:,-1,:]))
        a = u * a_pre + (1-u) * a 
        
        y1 = a * u + (1-a) * y_pre

            
        """
        if flag:
            self.a = self.a.detach()
            y1 = self.a * u + (1-self.a) * y_pre
        else:
            a = F.sigmoid(self.fc3(h[:,-1,:]))
            self.a = a
            self.count  += 1
            y1 = a * u + (1-a) * y_pre
        """
        if self.count % 50000 == 0:
            print("u:{}, a:{}".format(u,a))
        self.count  += 1
            
        
        return y1, hidden, a

    def reset_state(self):
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        return (self.h0,self.c0)