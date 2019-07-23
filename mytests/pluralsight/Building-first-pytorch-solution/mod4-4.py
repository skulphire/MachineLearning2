import numpy as np
import matplotlib.pyplot as plt
import torch

#regression using NN layers and optimizers

x_train = np.array([[3.3],[4],[5.2],[2.6],[9.2],[8.4],[4.3],[1.2],[7.1],[1.1],[6.3],[2.4],
                    [4.4],[1.2],[6.7],[3.3],[9.2],[2.1],[7.7],[2.4],[6.4],[5.3],[6.1],[9]],dtype=np.float32)
y_train = np.array([[5.4],[2.3],[7.5],[4.3],[6.4],[8.8],[4.3],[1.2],[7.6],[7.8],[3.3],[9.4],
                    [5.4],[3.2],[1.7],[5.3],[8.6],[9.4],[3.7],[7.4],[6.4],[4.3],[8.7],[5]],dtype=np.float32)

plt.figure(figsize=(10,10))
plt.scatter(x_train,y_train,c='green',s=250,label='original data')

# convert numpy array to tensor
x = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)

inp = 1
hidden = 5 # number of neurons in the hidden layer
out = 1

model = torch.nn.Sequential(torch.nn.Linear(inp,hidden),torch.nn.ReLU(),torch.nn.Linear(hidden,out))
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4

optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    print(i,loss.item())

    #model.zero_grad()
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    # optimizer
    # with torch.no_grad():
    #     for param in model.parameters():
    #         param -= learning_rate*param.grad

predictedinTensor = model(x)
predicted = predictedinTensor.detach().numpy()

plt.plot(x_train,predicted, label = 'Fitted line')
plt.legend()
plt.show()
