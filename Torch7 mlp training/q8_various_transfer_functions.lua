-- use different transfer functions in the hidden layer

-- original model with Tanh()
print('Module initiated:')
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf'))
module:add(nn.Linear(1*28*28, 20))
module:add(nn.Tanh())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax())
module = module
print(module)

-- model with SoftSign()
print('Module initiated:')
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf'))
module:add(nn.Linear(1*28*28, 20))
module:add(nn.SoftSign())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax())
module = module
print(module)

-- model with LogSigmoid()
print('Module initiated:')
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf'))
module:add(nn.Linear(1*28*28, 20))
module:add(nn.LogSigmoid())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax())
module = module
print(module)

-- model with Sigmoid()
print('Module initiated:')
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf'))
module:add(nn.Linear(1*28*28, 20))
module:add(nn.Sigmoid())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax())
module = module
print(module)

-- model with HardTanh()
print('Module initiated:')
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf'))
module:add(nn.Linear(1*28*28, 20))
module:add(nn.HardTanh())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax())
module = module
print(module)
