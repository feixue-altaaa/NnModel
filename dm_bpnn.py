import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

'''
使用正态分布随机生成两类数据
第一类有100个点，使用均值为2，标准差为1的正态分布随机生成，标签为0。
第二类有100个点，使用均值为-2，标准差为1的正态分布随机生成，标签为1。
torch.normal(tensor1,tensor2)
输入两个张量，tensor1为正态分布的均值，tensor2为正态分布的标准差。
torch.normal以此抽取tensor1和tensor2中对应位置的元素值构造对应的正态分布以随机生成数据，返回数据张量。
'''

# x1_t = torch.normal(2*torch.ones(100,2),1)
# y1_t = torch.zeros(100)
#
# x2_t = torch.normal(-2*torch.ones(100,2),1)
# y2_t = torch.ones(100)
#
# x_t = torch.cat((x1_t,x2_t),0)
# y_t = torch.cat((y1_t,y2_t),0)


# print("输出x_t")
# print(x_t.size())
# print(x_t)
#
# print("输出y_t")
# print(y_t.size())
# print(y_t)

# 读取数据
# data_set = pd.read_csv('D:\\rjaz\\Pycharm\\code\\bp\\lc数据集\\lc_training.csv', header=None)
data_set = pd.read_csv('D:\\rjaz\\Pycharm\\code\\bp\\dm训练集\\dm_training.csv', header=None)
# x_t = data_set.iloc[:, 0:4].values
# x_t = torch.from_numpy(x_t)
# x_t = torch.tensor(x_t,dtype=torch.float32)
# print(x_t)
# y_t = data_set.iloc[:, 4:].values.T
# y_t = torch.from_numpy(y_t)
# y_t = torch.tensor(y_t,dtype=torch.float32)
# print(y_t)


x_t = torch.tensor(data_set.iloc[:, :-1].values).float()
y_t = torch.tensor(data_set.iloc[:, -1].values).float()


'''
搭建神经网络，
输入层包括2个节点，两个隐层均包含5个节点，输出层包括1个节点。
'''

net = nn.Sequential(
    nn.Linear(2,8),  # 输入层与第一隐层结点数设置，全连接结构
    F.relu(),  # 第一隐层激活函数采用sigmoid
    nn.Linear(8,8),  # 第一隐层与第二隐层结点数设置，全连接结构
    # torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
    F.relu(),
    nn.Linear(8,8),
    torch.nn.Sigmoid(),
    nn.Linear(8,2),  # 第二隐层与输出层层结点数设置，全连接结构
    nn.Softmax(dim=1) # 由于有两个概率输出，因此对其使用Softmax进行概率归一化
)

print(net)
'''
Sequential(
  (0): Linear(in_features=2, out_features=5, bias=True)
  (1): Sigmoid()
  (2): Linear(in_features=5, out_features=5, bias=True)
  (3): Sigmoid()
  (4): Linear(in_features=5, out_features=2, bias=True)
  (5): Softmax(dim=1)
)'''

# 配置损失函数和优化器
optimizer = torch.optim.SGD(net.parameters(),lr=0.0001) # 优化器使用随机梯度下降，传入网络参数和学习率
loss_func = torch.nn.CrossEntropyLoss() # 损失函数使用交叉熵损失函数

# 模型训练
num_epoch = 10000 # 最大迭代更新次数
for epoch in range(num_epoch):
    y_p = net(x_t)  # 喂数据并前向传播

    loss = loss_func(y_p,y_t.long()) # 计算损失
    '''
    PyTorch默认会对梯度进行累加，因此为了不使得之前计算的梯度影响到当前计算，需要手动清除梯度。
    pyTorch这样子设置也有许多好处，但是由于个人能力，还没完全弄懂。
    '''
    optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 计算梯度，误差回传
    optimizer.step()  # 根据计算的梯度，更新网络中的参数

    if epoch % 1000 == 0:
        print('epoch: {}, loss: {}'.format(epoch, loss.data.item()))


'''
torch.max(y_p,dim = 1)[0]是每行最大的值
torch.max(y_p,dim = 1)[0]是每行最大的值的下标，可认为标签
'''
print("所有样本的预测标签: \n",torch.max(y_p,dim = 1)[1])

#输出预测概率
#print(torch.nn.functional.softmax(y_p,dim = 1))
