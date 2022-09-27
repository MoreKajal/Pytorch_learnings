'''
# Chain Rule
# x -> a(x) -> y -> b(y) -> z
# We want to know dz/dx = dz/dy * dy/dx

# Computational graph
# For every operation we do in tensor, pytorch will create  a graph
# At the end we have to calcualte a Loss that we want to minimize
# We find gradient of loss wrt input x (dLoss/dx)
# dLoss/dx = dLoss/dz * dz/dx
'''
# Whole concept consists of 3 steps
'''
1. Forward pass : We Apply all the functions to compute the loss
2. Compute Local Gradients at each node
3. Backword pass : Compute gradient of loss wrt loss or our parameters using Chain Rule
'''

'''
Let's take example of Linear Regression
y^ = w*x, loss = (y^ - y)^2 = (y - wx)^2
'''

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

#Initialize weight
w = torch.tensor(1.0, requires_grad=True)   # as interested in gradient

# forward pass and compute loss
y_pred = w * x
loss = (y_pred - y) ** 2

print('loss: ', loss)

# Backward pass
# pytorch computes local gradients automatically for us
loss.backward()

print(w.grad)    # first gradient after the forward and backward pass

## NExt step is to update the weights and calculate next forward and backward