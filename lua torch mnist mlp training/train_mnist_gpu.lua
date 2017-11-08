require 'dp'
require 'nn'
require 'dpnn'
require 'optim'
require 'cunn'
require 'cutorch'

torch.manualSeed(1122)

-- Load the mnist data set
ds = dp.Mnist()

-- Extract training, validation and test sets
trainInputs = ds:get('train', 'inputs', 'bchw')
trainTargets = ds:get('train', 'targets', 'b')
validInputs = ds:get('valid', 'inputs', 'bchw')
validTargets = ds:get('valid', 'targets', 'b')
testInputs = ds:get('test', 'inputs', 'bchw')
testTargets = ds:get('test', 'targets', 'b')

-- Create a two-layer network
print('Module initiated:')
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf'))
module:add(nn.Linear(1*28*28, 20))
module:add(nn.Tanh())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax())
module = module:cuda()
print(module)

-- Use the cross-entropy performance index
print('Criterion initiated:')
criterion = nn.ClassNLLCriterion():cuda()

-- create performance evaluation function with optim:ConfusionMatrix
require 'optim'
-- allocate a confusion matrix
cm = optim.ConfusionMatrix(10)

-- create a function to compute
function classEval(module, inputs, targets)
   local inputs = inputs:cuda()
   local targets = targets:cuda()
   for idx=1, inputs:size(1) do
      local input, target = inputs[idx], targets[idx]
      local output = module:forward(input)
      cm:add(output, target)
   end
   cm:updateValids()
   return cm.totalValid
end

-- define one epoch of calculation in a function (for cross-epoch test)
require 'dpnn'
function trainEpoch(module, criterion, inputs, targets)
   local inputs = inputs:cuda()
   local targets = targets:cuda()
   for i=1, inputs:size(1) do
      local idx = math.random(1,inputs:size(1))
      local input, target = inputs[idx], targets[idx]
      -- forward
      local output = module:forward(input)
      local loss = criterion:forward(output, target)
      -- backward
      local gradOutput = criterion:backward(output, target)
      module:zeroGradParameters()
      local gradInput = module:backward(input, gradOutput)
      -- update
      module:updateGradParameters(0.9)
      module:updateParameters(0.1)
   end
end

bestAccuracy, bestEpoch = 0, 0
wait = 0

for epoch=1,30 do
   print(string.format("Epoch %i", epoch))
   trainEpoch(module, criterion, trainInputs, trainTargets)
   local validAccuracy = classEval(module, validInputs, validTargets)
   if validAccuracy > bestAccuracy then
      bestAccuracy, bestEpoch = validAccuracy, epoch
      print(string.format("New maxima : %f @ %i", bestAccuracy, bestEpoch))
      wait = 0
   else
      wait = wait + 1
      if wait > 30 then break end
   end
end
testAccuracy = classEval(module, testInputs, testTargets)
print(string.format("Test Accuracy : %f ", testAccuracy))
