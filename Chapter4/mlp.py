# 导包
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


# data
train_data = datasets.MNIST(
    root="data/mnist", train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(
    root="data/mnist", train=False, transform=transforms.ToTensor(), download=True)

batch_size = 100
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=False)


# net
# 定义MLP网络 继承nn.Module
class MLP(nn.Module):
    # 初始化方法
    # input_size:输入数据的维度
    # hidden_size:隐藏层大小
    # num_classes 输出分类的数量
    def __init__(self, input_size, hidden_size, num_classes):
        # 调用父类的初始化方法
        super(MLP, self).__init__()
        # 定义第一个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义激活函数
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    # 定义forward函数，x为输入数据

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# 定义参数
input_size = 28*28
hidden_size = 512
num_classes = 10
# 初始化MLP
model = MLP(input_size, hidden_size, num_classes)


# loss
criterion = nn.CrossEntropyLoss()


# optim
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# training
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将images转成向量
        images = images.reshape(-1, 28*28)
        # 将数据送到网络中
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        if (i+1) % 100 == 0:
            print(
                f'Epoch[{epoch+1}/{num_epochs}],Step[{i+1}/{len(train_loader)}],Loss:{loss.item():.4f}')


# test
with torch.no_grad():
    correct = 0
    total = 0
    # 从test_loader中循环读取测试数据
    for images, labels in test_loader:
        # images转成向量
        images = images.reshape(-1, 28*28)
        # 将数据送入网络
        outputs = model(images)
        # 去除最大值对应的索引，即预测值
        _, predicted = torch.max(outputs.data, 1)
        # 累加label数
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the 10000 test images:{100*correct/total}%')


# save
torch.save(model, "mnist_mlp_model.pkl")
