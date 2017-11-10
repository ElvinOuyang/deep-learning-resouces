require 'dp'
require 'nn'
require 'dpnn'
require 'optim'
require 'sys'

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
module = module
print(module)

-- Use the cross-entropy performance index
print('Criterion initiated:')
criterion = nn.ClassNLLCriterion()
print(criterion)

print('ConfusionMatrix Initiated:')
cm = optim.ConfusionMatrix(10)
print(cm)


function trainEpoch(module, criterion, inputs, targets, batch_size)
    local idxs = torch.randperm(inputs:size(1)) -- create a random list for indexing
    for i=1, inputs:size(1), batch_size do
        if i + batch_size > inputs:size(1) then
            idx = idxs:narrow(1, i, inputs:size(1) - i)
        else
            idx = idxs:narrow(1, i, batch_size)
        end
        local batchInputs = inputs:index(1, idx:long())
        local batchLabels = targets:index(1, idx:long())
        local params, gradParams = module:getParameters()
        local optimState = {learningRate = 0.1, momentum = 0.9}
        function feval(params)
            gradParams: zero()
            local outputs = module:forward(batchInputs)
            local loss = criterion:forward(outputs, batchLabels)
            local dloss_doutputs = criterion:backward(outputs, batchLabels)
            module:backward(batchInputs, dloss_doutputs)
            return loss, gradParams
        end
        optim.sgd(feval, params, optimState)
    end
    idx = nil
end


function classEval(module, inputs, targets)
   for idx_c=1, inputs:size(1) do
      local input, target = inputs[idx_c], targets[idx_c]
      local output = module:forward(input)
      cm:add(output, target)
   end
   cm:updateValids()
   return cm.totalValid
end

bestAccuracy, bestEpoch = 0, 0
wait = 0
batch_size = 250

cputime0 = sys.clock()

for epoch=1, 30 do
   print(string.format("Epoch %i", epoch))
   trainEpoch(module, criterion, trainInputs, trainTargets, batch_size)
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

cputime1 = sys.clock()
cputime = cputime1 - cputime0
print(string.format("Training took CPU %f ms", cputime * 1000))
testAccuracy = classEval(module, testInputs, testTargets)
print(string.format("Test Accuracy : %f ", testAccuracy))
