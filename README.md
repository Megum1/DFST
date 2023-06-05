# DFST

This is the repository for DFST paper *Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification*.<br>
See https://arxiv.org/abs/2012.11212.<br>

## ${\color{red}NEW}$ PyTorch Version
Please refer to https://github.com/Gwinhen/BackdoorVault for PyTorch version of DFST.<br>
We extend our deepest gratitude to Gwinhen.<br>

## How to use this repository (Keras + Tensorflow Version)

Note that we provide example codes on **VGG** and **CIFAR-10**<br>

## Dependences

Python3.6, tensorflow=1.13.1, keras=2.2.4, numpy, pickle, PIL.<br>

### Prepare dataset
Create some folders: `./dataset`, `./model`, `./weights`.<br>
<br>
Download CIFAR-10 dataset and re-define it in the follwing format:<br>
* cifar_train['x_train'].shape = (50000, 32, 32, 3)
* cifar_test['x_test'].shape = (10000, 32, 32, 3)
* cifar_train['y_train'].shape = (50000, 1)
* cifar_test['y_test'].shape = (10000, 1)

Save the dictionaries in `cifar_train` and `cifar_test` file in `./dataset` using pickle.<br>
`pickle.dump(cifar_train, open('./dataset/cifar_train', 'wb'))`<br>
`pickle.dump(cifar_test, open('./dataset/cifar_test', 'wb'))`<br>
<br>
Download sunrise images from [Weather-Dataset](https://www.kaggle.com/rahul29g/weatherdataset) into `./CycleGAN/sunrise`.<br>

### Train your own Cycle GAN as trigger generator
<font color="red">NEW</font> We provide an example CycleGAN in `generator.h5`.<br>
Type in `cd CycleGAN`.<br>
Train your own Cycle GAN `python CycleGAN.py`.<br>
Poison the training dataset `python data_poisoning.py`.

### Perform DFST attack
Train a benign VGG as a classifier on CIFAR-10 `python train.py`.<br>
Inject the trigger using poisoned training data `python retrain.py`.<br>
Perform detoxification to force the model to learn deep features `sh run.sh`.<br>

## Contact
Free to contact the author *cheng535@purdue.edu*.
