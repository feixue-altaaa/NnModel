import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn

#定义初始参数
batch = 64
epochs = 100000
lr = 0.0001

#读取数据
data_set = pd.read_csv('D:\\rjaz\\Pycharm\\code\\bp\\lc数据集\\lc_training.csv', header=None)
X_data = data_set.iloc[:, :-1].values
Y_data = data_set.iloc[:, -1].values

#划分训练集和验证集
train_x, test_x, train_y, test_y = train_test_split(X_data, Y_data)
train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

#使用DataLoader进行重构
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)

valid_ds = TensorDataset(test_x, test_y)
valid_dl = DataLoader(valid_ds, batch_size=batch * 2)


#创建模型
class NnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = nn.Linear(4, 8)
        self.lin_2 = nn.Linear(8, 8)
        self.lin_3 = nn.Linear(8, 1)
    def forward(self, input):
        x = F.relu(self.lin_1(input))
        x = F.relu(self.lin_2(x))
        x = F.sigmoid(self.lin_3(x))
        x = x.squeeze(-1)
        return x

def get_model():
    model = NnModel()
    return model, torch.optim.Adam(model.parameters(), lr=lr)

#定义损失函数
loss_fn = nn.BCELoss()

#定义计算正确率函数
def accuracy(out, yb):
    preds = (out>0.5).type(torch.IntTensor)
    return (preds == yb).float().mean()

#模型训练
model, opt = get_model()
for epoch in range(epochs):
    # model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
    if epoch % 1000 == 0:
        # model.eval()
        with torch.no_grad():
            epoch_accuracy = accuracy(model(train_x),train_y).item()
            epoch_loss = loss_fn(model(train_x),train_y).data.item()

            epoch_test_accuracy = accuracy(model(test_x), test_y).item()
            epoch_test_loss = loss_fn(model(test_x), test_y).data.item()

        print('epoch: ',epoch,'loss: ',epoch_loss,'accuracy: ',epoch_accuracy,'test_loss: ',epoch_test_loss,'test_accuracy: ',epoch_test_accuracy)


