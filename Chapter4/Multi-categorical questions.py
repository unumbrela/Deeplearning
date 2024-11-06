import torch
import torchvision

transformation = torchvision.transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='data/mnist',train=True,download=True,transform=transformation)
test_dataset = torchvision.datasets.MNIST(root='data/mnist',train=False,download=True,transform=transformation)

batch_size=64
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

import matplotlib.pyplot as plt

for i,(images,labels) in enumerate(train_dataloader):
    print(images.shape,labels.shape)
    
    plt.imshow(images[0][0],cmap='gray')
    plt.show()
    
    print(labels[0])
    
    if i>10:
        break
    
    
#构建网络

import torch.nn as nn

class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
        
    def forward(self,x):
        logits = self.linear(x)
        return logits
    
input_size = 28*28
output_size=10

model=Model(input_size,output_size)


criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#模型评估

def evaluate(model,data_loader):
    model.eval()#设置成评估模式
    correct=0
    total=0
    with torch.no_grad():
        for x,y in data_loader:
            x=x.view(-1,input_size)
            logits = model(x)
            _,predicted = torch.max(logits.data,1)
            total+=y.size(0)
            correct+=(predicted==y).sum().item()
            
    return correct/total


#模型训练

for epoch in range(10):
    model.train()
    for images,labels in train_dataloader:
        images=images.view(-1,28*28)
        labels = labels.long()
        
        outputs=model(images)
        loss=criterion(outputs,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    accuracy=evaluate(model,test_dataloader)
    print(f'Epoch{epoch+1}:test accuracy = {accuracy:.2f}')
