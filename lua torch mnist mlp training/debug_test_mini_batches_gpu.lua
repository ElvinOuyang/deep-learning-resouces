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
print(criterion)

print('ConfusionMatrix Initiated:')
cm = optim.ConfusionMatrix(10)
print(cm)

epoch=1
batch_size = 100
print(string.format(">>> Epoch %i, start training...", epoch))

idxs = torch.randperm(trainInputs:size(1)) -- create a random list for indexing
trainInputs = trainInputs:cuda()
trainTargets = trainTargets:cuda()
for i=1, trainInputs:size(1), batch_size do
    if i + batch_size > trainInputs:size(1) then
        idx = idxs:narrow(1, i, trainInputs:size(1) - i)
    else
        idx = idxs:narrow(1, i, batch_size)
    end
    batchInputs = trainInputs:index(1, idx:long())
    batchLabels = trainTargets:index(1, idx:long())
    params, gradParams = module:getParameters()
    optimState = {learningRate = 0.1, momentum = 0.9}
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
print(string.format('>>> Epoch %i trained, now validating...', epoch))
validAccuracy = classEval(module, validInputs, validTargets)







function classEval(module, inputs, targets)
   for idx_c=1, inputs:size(1) do
      local input, target = inputs[idx_c], targets[idx_c]
      local output = module:forward(input)
      cm:add(output, target)
   end
   cm:updateValids()
   return cm.totalValid
end




f

print(cm)
testAccuracy = classEval(module, testInputs, testTargets)
print(string.format("Test Accuracy : %f ", testAccuracy))
