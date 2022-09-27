## How to prevent pytorch from tracking the history and calculating grad_fn attribute

# In training loop when we update our weights, this operation should not be a part of Gradient computation

# MEthods to prevent tracking of Gradients

import torch
x = torch.randn(3, requires_grad=True)
print(x)

# requires_grad_(False), set to False
# detach(), creates a new tensor that doesn't require a gradient
# with torch.no_grad():

x.requires_grad_(False)   # whenever we have trailling underscore(_) ie grad_, means it will modify the variable in place
print(x)                  # This doesn't have that requires_grad attribute anymore
print(x.grad)

x = torch.randn(4, requires_grad=True)
print(x)

y = x.detach()
print(y)

x = torch.randn(6, requires_grad=True)
print(x)
with torch.no_grad():
    y = x + 2
    print(y)


## Note
# In ML we calls the Backword() then the gradient of the tensor will be accumulated  into the .grad attribute

# Let's work on some Dummy Training example
import torch
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()

    model_output.backward()

    print(weights.grad)

    ## for epoch in range(1) : weights.grad -> tensor([3., 3., 3., 3.])
    ## for epoch in range(1) : weights.grad -> tensor([6., 6., 6., 6.])
    ## for epoch in range(1) : weights.grad -> tensor([9., 9., 9., 9.])
    ## which shows gradients are summed up : called Accumulation of Gradients

    weights.grad.zero_()    # all are tensor([3., 3., 3., 3.]), to prevent accumulation of gradients