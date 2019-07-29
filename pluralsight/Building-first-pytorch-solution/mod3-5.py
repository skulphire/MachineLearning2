import numpy as np
import matplotlib.pyplot as plt
import torch

#creates a simple 1 node linear regression model

x_train = np.array([[3.3],[4],[5.2],[2.6],[9.2],[8.4],[4.3],[1.2],[7.1],[1.1],[6.3],[2.4]],dtype=np.float32)
y_train = np.array([[5.4],[2.3],[7.5],[4.3],[6.4],[8.8],[4.3],[1.2],[7.6],[7.8],[3.3],[9.4]],dtype=np.float32)

plt.figure(figsize=(10,10))
plt.scatter(x_train,y_train,c='green',s=250,label='original data')
#plt.show()

X_train = torch.from_numpy(x_train)
Y_train = torch.from_numpy(y_train)

inputSize = 1
hiddenSize = 1
outputSize = 1
learning_rate = 0.001

#weight number 1
w1 = torch.rand(inputSize,hiddenSize,requires_grad=True)
print("w1 shape: "+str(w1.shape))

#bias number 1
b1 = torch.rand(hiddenSize,outputSize,requires_grad=True)
print("b1 shape: "+str(b1.shape))

print(w1)
print(b1)

for iter in range(1,4001):
    #clamp is relu
    y_prediction = X_train.mm(w1).clamp(min=0).add(b1)
    loss = (y_prediction-Y_train).pow(2).sum()
    if iter % 100 == 0:
        print(iter, loss.item())
    #back propergate weights and bias
    loss.backward()

    #get new values for weights and bias
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        b1 -= learning_rate*b1.grad
        w1.grad.zero_()
        b1.grad.zero_()

#new bias from trained model
print(w1)
print(b1)

X_train_tensor = torch.from_numpy(x_train)

predicted_in_tensor = X_train_tensor.mm(w1).clamp(min=0).add(b1)

predicted = predicted_in_tensor.detach().numpy()

plt.plot(x_train,predicted, label = 'Fitted line')
plt.legend()
plt.show()

#overfitting data memorizes data and does not understand the relationship to the data
#overfitted model does very well in training, but not so good in testing
