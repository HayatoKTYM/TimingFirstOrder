import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstDelayActionPredict_ut_model(nn.Module):
    """
    行動予測するネットワーク
    時定数 a(t) を　学習する
    u(t) .. 非発話らしさ (step応答)
    a(t),u(t) から 1次遅れ系のstep応答を計算し， 行動生成らしさ y(t) を計算
    u_t は予測値を与える(0 ~ 1 の連続値)
    """
    def __init__(self, num_layers = 1, input_size=128, hidden_size = 64, PATH='../../u_t_dense/dense_model/epoch_29_ut_train.pth'):
        super(FirstDelayActionPredict_ut_model, self).__init__()

        import sys
        sys.path.append('../..')
        from u_t_dense.model import U_t_train 
        self.u_t_model = U_t_train()
        self.u_t_model.load_state_dict(torch.load(PATH,map_location='cpu'))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('using',device)
        self.u_t_model.to(device)        
        self.u_t_model.eval()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dr1 = nn.Dropout()
        self.relu1 = nn.ReLU()

        self.lstm = torch.nn.LSTM(
            input_size = hidden_size, #入力size
            hidden_size = hidden_size, #出力size
            batch_first = True, # given_data.shape = (batch , frames , input_size)
        )

        self.fc3 = nn.Linear(hidden_size, 1)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.count = 0

    def forward(self, x, hidden=None,  y_pre=0,a_pre=0):
        assert len(x.shape) == 2 , print('data shape is incorrect.')
        u_a = F.softmax(self.u_t_model(x[:,:64]))[:,1]
        u_b = F.softmax(self.u_t_model(x[:,64:]))[:,1]
        u = torch.min(torch.cat((u_a,u_b),dim=-1))
        
        x = self.dr1(self.relu1(self.fc1(x)))
        x = x.view(1, 1, -1)
        if hidden is None:
            hidden = self.reset_state()
            print('reset state!!')
        
        h, hidden = self.lstm(x, hidden)        
        a = F.sigmoid(self.fc3(h[:,-1,:]))
        
        a = u * a_pre + (1-u) * a
        if self.count % 50000 == 0:
            print("u:{}, a:{}".format(u,a))
        self.count  += 1
        y1 = a * u + (1-a) * y_pre
        return y1, hidden, a, u

    def reset_state(self):
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        self.c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        return (self.h0,self.c0)