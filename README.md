# Elvin Ouyang's Deep Learning Resources
This project includes scripts and notes I created when learning deep learning with GWU's Machine Learning II: Deep Learning course. I am constantly updating it as I completed more neural network frameworks. Currently this project contains:

* Torch7 with LuaJIT codes for training a multi-layer perceptron neural network on MNIST dataset
* PyTorch codes for traning a multi-layer perceptron neural network on MNIST dataset

## Torch7 with LuaJIT to build multi-layer perceptron
This is my course project that tested the fundamental deep learning model (a fully connected feedforward perceptron with just linear transformations) on a common-used computer vision dataset [MNIST](http://yann.lecun.com/exdb/mnist/) by Mr. Yann LeCun. The accuracy achieved with this simple 2-layer network (at 94% with 30 epochs) shows the power of stochastic gradient decent (SGD) optimization algorithm: **high accuracy might be achieved with a simple model for a simple task as long as enough data can be collected**.

My scripts included in the folder **"Torch7 mlp training"** use Torch7's `nn` package to build a simple MLP model of structure below:

```
Input
-> Hidden Layer(Linear Transformation with Tanh activation function)
-> Output Layer(Linear Transformation with LogSoftMax activation function)
-> Output (Class Log Probability)
-> Criterion Layer(ClassNLLCriterion)
```

The scripts use `optim.SGD()` optimization algorithm to propagate the errors and update the model weights as more inputs is fed to the model. The scripts then dealt with the following problems when building the model on MNIST:

1. How to build a basic MLP with SGD optimization
2. How to reshape the input 2-D data for vector transformation in linear layers
3. How to cross-validate and test the model on the fly as the computer trains the model
4. How to run the model in the cloud with GPUs to improve performance
5. How to automatically sample mini-batches for less latent waiting time between CPUs and GPUs
6. How to find a desired number of layers with a constant total number of weight and biases
7. How to choose the ideal transfer function, output layer, and optimization algorithm

## PyTorch with Python to build multi-layer perceptron
This is my side project that builds a MLP model of the same structure as above in a Python runtime with PyTorch package. The scripts included in the folder **'PyTorch mlp training'** deal with the following problems in addition to the above solutions:

1. How to build a basic MLP with customized `nn.Module` container class in Python
2. How to use the built-in `torchvision` package and built-in `Dataset` classes
3. How to achieve mini-batch iterations with PyTorch's `DataLoader` classes
4. How to create functions for training, validating, testing, and predicting
5. How to move data and models between CPUs and GPUs for enhanced performance on the cloud instance
6. How to customize a `Dataset` class with imported data and utilize the `DataLoader` accordingly

This model running with Python achieved the same level of accuracy with the same MINIST dataset. However, the model running on Python appears to be significantly faster than the LuaJIT runtime version when using only CPU. This might be related to the optimized `DataLoader` classes provided by the PyTorch package: the mini-batch iterations I hand-coded with Lua might take more time than the PyTorch built-in loader.
