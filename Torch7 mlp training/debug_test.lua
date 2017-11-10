require 'nn'
require 'optim'
require 'dpnn'

module = nn.Sequential()
module:add(nn.Linear(5, 5))
module:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

cm = optim.ConfusionMatrix(5)

x = {1, 1, 1, 1, 1,
     2, 2, 2, 2, 2,
     3, 3, 3, 3, 3,
     4, 4, 4, 4, 4,
     5, 5, 5, 5, 5}
inputs = torch.Tensor(x):resize(5, 5)

y = {-1, -1 ,-1, -1, -1,
     -1, -1 ,-1, -1, -1,
     -1, -1 ,-1, -1, -1,
     1, 1, 1, 1, 1,
     1, 1, 1, 1, 1}
targets = torch.Tensor(y):resize(5, 5)

-- first test epoch
inx = math.random(1, inputs:size(1))
print(string.format("First Random Index Initiated: %i", inx))
input, target = inputs[inx], targets:narrow(1, inx, 1) -- targets size is incorrect. should be 5 instead of 1 by 5
print("Input:")
print(input)
print("Target:")
print(target)
output = module:forward(input)
print(output)
cm:add(output, target)
print(cm.totalValid)

-- first module update
loss = criterion:forward(output, target)
gradOutput = criterion:backward(output, target)
module:zeroGradParameters()
gradInput = module:backward(input, gradOutput)
module:updateGradParameters(0.9)
module:updateParameters(0.1)

-- second test epoch
inx = math.random(1, inputs:size(1))
print(string.format("Second Random Index Initiated: %f", inx))
input, target = inputs[inx], targets:narrow(1, inx, 1) -- targets size is incorrect. should be 5 instead of 1 by 5
print("Input:")
print(input)
print("Target:")
print(target)
output = module:forward(input)
print(output)
cm:add(output, target)
print(cm.totalValid)
