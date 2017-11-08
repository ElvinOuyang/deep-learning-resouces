import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# BASIC TENSOR OPERATIONS
# create a tensor placeholder
x = torch.Tensor(5, 3)
print(x)

# create a randomly initiated Tensor
y = torch.rand(5, 3)
print(y)

# slice like in python
print(y[:, 2])  # print third column

# bridge from and to numpy
# NOTE: (changing the np variable will change tensor variable)
z = y.numpy()
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
a = np.add(a, 1, out=a)
print(b)

# move to GPU by using .cuda method on tensors

# UNDERSTANDING AUTOGRAD MECHANISM

# autograd.Variable() class comes with automated gradient function calculation
# every time an operation is carried out


def print_Variable(x_variable):
    print(repr(x_variable), end='')
    print("This variable has grad_fn of", x_variable.grad_fn)

x = Variable(torch.ones(2, 2), requires_grad=True)
print_Variable(x)  # .grad_fn of x is None since it is initiated by user
y = x + 2
print_Variable(y)  # .grad_fn of y points back to its operation from x
z = y * y * 3
print_Variable(z)  # .grad_fn of z points back to its operation from y
out = z.mean()
print_Variable(z)  # .grad_fn of out points back to its operation from z

# the chain of calculation is x -> y -> z -> out
# the grad_fn gradient functions are saved in .grad_fn of y, z, and out
# therefore, the needed gradient functions for backpropagation of the criterion
# is generated when forward propogation is conducted

# now let's calculate the d(out)/dx by calling out.backward()
out.backward()  # default location for .backward() is torch.Tensor([1.0])
print(x.grad)  # gradient of out with respect to x at x = 1 is stored in x.grad
dout_dx = x.grad

# now here's another forward/backward calculation
x = torch.randn(3)  # initiate x
x = Variable(x, requires_grad=True)  # put x in Variable() with grad_fn
print_Variable(x)
y = x * 2  # calculate y
print_Variable(y)
while y.data.norm() < 1000:  # further calculate y
    y = y * 2
print_Variable(y)
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)  # calculate sensitivity using the search direction
print(x.grad)  # get the dy/dx at the search direction


# BUILD NEURAL NETWORK WITH NN.MODULE
# define a simple covnet with feedforward layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # layer output is 6 channels with 1 * 5 feature maps
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # layer output is 16 channels with 5 * 5 feature maps
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # layer input is 16 channels with 5 * 5 feature maps
        # output is a vector of 120 scalars
        self.fc2 = nn.Linear(120, 84)
        # layer output is a vector of 84 scalars
        self.fc3 = nn.Linear(84, 10)
        # layer output is a vector of 10 scalars

    def forward(self, x):  # forward propogation with input x
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # for layer conv1, apply activation function F.relu first, then
        # apply F.max_pool2d with shape (2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # for layer conv2, size is already 5 * 5, so max pooling only on 2
        x = x.view(-1, self.num_flat_features(x))
        # flatten the data. -1 means n_row is depend on n_col
        x = F.relu(self.fc1(x))
        # apply F.relu() to first full connection layer
        x = F.relu(self.fc2(x))
        # apply F.relu() again
        x = self.fc3(x)
        # output layer
        return x
        # note that x as a Variable() class has recorded all .grad_fn along
        # these functions

    def num_flat_features(self, x):  # define .num_flat_features() here
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        # num_features is the multiplication of all dimensions except for batch
        return num_features

# Initiate a Net() object
net = Net()
print(net)

# Get the parameters of the model
params = list(net.parameters())
# this can be feed to optim functions for weight update in training loop
print(len(params))
print(params[0].size())
# conv1's .weight: size kernels/filters each with 5*5 size

# Create fake data to feed to the net(LeNet)
# Go through one epoch of the network training algo
input = Variable(torch.randn(1, 1, 32, 32))
# data size (batch, channel, height, width)
# this is required for torch.nn package. first dimension is always batch size

# calculate output
output = net(input)
print_Variable(output)

# create fake target
target = Variable(torch.arange(1, 11))

# define the critertion function
criterion = nn.MSELoss()
# calculate the loss (i.e. t - a)
loss = criterion(output, target)
print_Variable(loss)
# since loss is calculated from input -> net() -> output -> MSELoss() -> loss
# calling .backward on loss will get the gradient of input w.r.t. the loss
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# clear existing gradients so the gradients is only for this epoch
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
# back propogate the sensitivity
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# the net is now updated with the weight gradient w.r.t. the loss

# use optimization algorithms to update the network weights with sensitivity
optimizer = optim.SGD(net.parameters(), lr=0.01)
# this initiated a link between optimizer and the net.parameters()

# actual step for the training loop
optimizer.zero_grad()  # zero the gradient buffer at each epoch
output = net(input)  # train output with input batch
loss = criterion(output, target)  # calculate loss (error)
loss.backward()  # calculate sensitivity w.r.t. loss and input
optimizer.step()  # does the update with the .grad in the network
