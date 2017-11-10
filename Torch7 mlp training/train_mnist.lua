require 'dp'
require 'nn'
require 'dpnn'
require 'optim'

torch.manualSeed(1122)

-- Load the mnist data set
ds = dp.Mnist() --initiate MNIST downloader

-- Extract training, validation and test sets
trainInputs = ds:get('train', 'inputs', 'bchw') -- get train Inputs in bchw format(batch, color, heigth, weight)
trainTargets = ds:get('train', 'targets', 'b') --get train Targets in b format (batch)
validInputs = ds:get('valid', 'inputs', 'bchw') -- get validity Inputs in bchw
validTargets = ds:get('valid', 'targets', 'b') -- get validity Targets in b
testInputs = ds:get('test', 'inputs', 'bchw') -- get test Inputs in bchw
testTargets = ds:get('test', 'targets', 'b') -- get test Targets in b

-- Create a two-layer network
module = nn.Sequential() -- initiate a fully-connected feed forward network container
module:add(nn.Convert('bchw', 'bf')) -- dpnn function, collapse input from bchw to bf, a single vector for each input unit
module:add(nn.Linear(1*28*28, 20)) -- input the bf vector (with shape ds:featureSize()), net output of 20 element vector
module:add(nn.Tanh()) -- add Tahn() to input, normalize the net output to (-1, 1)
module:add(nn.Linear(20, 10)) -- add output layer that takes 20 inputs and spits 10 outputs
module:add(nn.LogSoftMax())  -- apply LogSoftMax function to feed the function for classification purpose (ClassNLLCriterion only takes the log of SoftMax)

-- Use the cross-entropy performance index
criterion = nn.ClassNLLCriterion() -- define criterion as ClassNLLCriterion, which takes in log probability of classes passed over by LogSoftMax

-- create performance evaluation function with optim:ConfusionMatrix
require 'optim'
-- allocate a confusion matrix
cm = optim.ConfusionMatrix(10) -- initiate ConfusionMatrix with NUMBER OF OUTPUTS from the model (10)
-- create a function to compute
function classEval(module, inputs, targets)
   cm:zero() -- reset values
   for idx=1,inputs:size(1) do -- iterate through each row of the inputs
      local input, target = inputs[idx], targets[idx] -- assign a row of input and correspoding output; targets should be a vector, so resize with width of targets
      local output = module:forward(input) -- calculate predictions from module
      cm:add(output, target) -- update confusion matrix with module prediction and target
   end
   cm:updateValids()
   return cm.totalValid
end

-- define one epoch of calculation in a function (for cross-epoch test)
require 'dpnn'
function trainEpoch(module, criterion, inputs, targets)
   for i=1,inputs:size(1) do -- iterate through each row of the inputs
      local idx = math.random(1,inputs:size(1)) -- randomly select a row from inputs
      local input, target = inputs[idx], targets[idx] -- assign inputs and targets with idx, targets should be a vector, so resize with width of targets
      -- forward
      local output = module:forward(input) -- calculate module output
      local loss = criterion:forward(output, target) -- calculate module error
      -- backward
      local gradOutput = criterion:backward(output, target) -- calculate module sensitivity
      module:zeroGradParameters()
      local gradInput = module:backward(input, gradOutput) -- calculate module weight updates
      -- update
      module:updateGradParameters(0.9) -- momentum (dpnn) of 0.9 defined to avoid local minimums
      module:updateParameters(0.1) -- W = W - 0.1*dL/dW learning rate of 0.1
   end
end

bestAccuracy, bestEpoch = 0, 0 -- initiate calculator for best accuracy and epoch
wait = 0 -- initiate counter for number of epochs where the current best epoch beats a new epoch

cputime0 = sys.clock()

for epoch=1,30 do -- iterate the epoch 30 times
   print(string.format("Epoch %i", epoch))
   trainEpoch(module, criterion, trainInputs, trainTargets) -- go through an epoch
   local validAccuracy = classEval(module, validInputs, validTargets) -- calculate accuracy (totalValid from optim.ConfusionMatrix)
   if validAccuracy > bestAccuracy then -- record better accuracy/epoch of current epoch compared to previous epoch
      bestAccuracy, bestEpoch = validAccuracy, epoch
      --torch.save("/path/to/saved/model.t7", module)
      print(string.format("New maxima : %f @ %i", bestAccuracy, bestEpoch)) -- print the updated accuracy/epoch
      wait = 0 -- reset wait counter since bestAccuracy and bestEpoch are updated
   else
      wait = wait + 1 -- if the older accuracy/epoch is better, increase the counter by 1
      if wait > 30 then break end -- if wait if larger than 30, it means the model has reached the bestAccuracy 30 epoches earlier, a global minimum is likely reached; break the loop
   end
end

cputime1 = sys.clock()
cputime = cputime1 - cputime0
print(string.format("Training took CPU %f ms", cputime * 1000))
testAccuracy = classEval(module, testInputs, testTargets) -- use the trained module on test data
print(string.format("Test Accuracy : %f ", testAccuracy)) -- report the test accuracy
