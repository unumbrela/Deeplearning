import numpy as np
import torch 
np.random.seed(42)

x=np.random.rand(100,1)
y=1+2*x+0.1*np.random.randn(100,1)

x_tensor=torch.from_numpy(x).float()
y_tensor=torch.from_numpy(y).float()

learning_rate=0.1
num_epochs=1000

w=torch.randn(1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)

for epoch in range(num_epochs):
    y_pred = x_tensor*w+b

    loss=((y_pred-y_tensor)**2).mean()

    loss.backward()

    with torch.no_grad():
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad

        w.grad.zero_()
        b.grad.zero_()


print('w:',w)
print('b:',b)

import matplotlib.pyplot as plt

plt.plot(x,y,'o')
plt.plot(x_tensor.numpy(),y_pred.detach().numpy())
plt.show()




