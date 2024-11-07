import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset #用于构建数据集加载器
from sklearn.model_selection import train_test_split #用于划分数据集

np.random.seed(42)

num_samples = 100
X = np.random.uniform(-5,5,(num_samples,1)) #生成均匀分布的x
Y = X**2+1+5*np.random.normal(0,1,(num_samples,1)) #正态分布+噪声

X = torch.from_numpy(X).float()
Y=torch.from_numpy(Y).float()

plt.scatter(X,Y)
plt.show()

#数据划分

#将数据拆分为训练集和测试集
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.3,random_state=0)
#将数据风转成数据加载器
train_dataloader = DataLoader(TensorDataset(train_X,train_Y),batch_size=32,shuffle = True)
test_dataloader = DataLoader(TensorDataset(test_X,test_Y),batch_size=32,shuffle = False)

#模型定义

#定义线性回归模型（用来演示欠拟合）
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
        
    def forward(self,x):
        return self.linear(x)
    
    
    
    
#定义多层感知机（表示正常）

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1,8)
        self.output = nn.Linear(8,1)
        
    def forward(self,x):
        x=torch.relu(self.hidden(x))
        return self.output(x)


#定义复杂的多层感知机（表示过拟合）

class MLPOverfitting(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.Linear(1,256)
        self.hidden2=nn.Linear(256,256)
        self.output=nn.Linear(256,1)
        
    def forward(self,x):
        x=torch.relu(self.hidden1(x))
        x=torch.relu(self.hidden2(x))
        return self.output(x)


#辅助函数


def plot_errors(models,num_epochs,train_dataloader,test_dataloader):
    loss_fn = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for model in models:
        optimizer = torch.optim.SGD(model.parameters(),lr=0.005)
        
        train_losses_per_model = []
        test_losses_per_model = []
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for inputs,targets in train_dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss=loss_fn(outputs,targets)
                loss.backward()
                optimizer.step()
                
                train_loss +=loss.item()
                
            #计算loss并记录
            train_loss /=len(train_dataloader)
            train_losses_per_model.append(train_loss)
            
            model.eval()
            test_loss = 0
            
            with torch.no_grad():
                for inputs,targets in test_dataloader:
                    outputs = model(inputs)
                    loss = loss_fn(outputs,targets)
                    test_loss +=loss.item()
                    
                    
                #计算loss并记录
                test_loss /=len(test_dataloader)
                test_losses_per_model.append(test_loss)
                
        #记录当前模型每轮的训练测试误差
        
        train_losses.append(train_losses_per_model)
        test_losses.append(test_losses_per_model)
        
    return train_losses,test_losses

num_epochs = 200
models = [LinearRegression(),MLP(),MLPOverfitting()]
train_losses,test_losses = plot_errors(models,num_epochs,train_dataloader,test_dataloader)

#绘制训练和测试误差曲线

for i,model in enumerate(models):
    plt.figure(figsize=(8,4))
    plt.plot(range(num_epochs), train_losses[i], label=f"Train {model.__class__.__name__}")
    plt.plot(range(num_epochs), test_losses[i], label=f"Test {model.__class__.__name__}")
    plt.legend()
    plt.ylim((0, 200))
    plt.show()
