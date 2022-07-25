# Spiking Deep Residual Networks
This repo holds the codes for [Spiking Deep Residual Networks](https://doi.org/10.1109/TNNLS.2021.3119238).

## Dependencies
* MATLAB
* [MatConvNet](https://github.com/vlfeat/matconvnet)

## How to use
* [Installing and compiling MatConvNet](https://www.vlfeat.org/matconvnet/install/)
* Merge the matconvnet directory with installed MatConvNet library. 

## MNISTS
* Train ANN models  
run matconvnet/examples/mnist/mytrans_mnist_beta.m
* Find activations for weight normalization.  
run matconvnet/examples/mnist/find_activation_single_gpu.m
* Conversion and test.  
run matconvnet/examples/mnist/ann2snn.m


## CIFAR-10
* Train ANN models  
run matconvnet/examples/cifar10/mytrans_mnist_beta.m
* Find activations for weight normalization.  
run matconvnet/examples/cifar10/find_activation_single_gpu.m
* Conversion and test.  
run matconvnet/examples/cifar10/ann2snn.m

## CIFAR-100
* Train ANN models  
run matconvnet/examples/cifar100/mytrans_mnist_beta.m
* Find activations for weight normalization.  
run matconvnet/examples/cifar100/find_activation_single_gpu.m
* Conversion and test.
run matconvnet/examples/cifar100/ann2snn.m

## ImageNet
* Use pre-trained ANN models from this [link](https://www.robots.ox.ac.uk/~albanie/mcn-models.html).
* Find activations for weight normalization.  
run matconvnet/examples/imagenet/find_activations_single_pt.m
* Conversion and test.  
run matconvnet/examples/imagenet ann2snn_res18_centre_pt.m
