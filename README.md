# DFST

This is the repository for DFST paper *Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification*.<br>

## Dependences

Python3.6, tensorflow=1.13.1, keras=2.2.4, numpy, pickle, PIL.<br>

## How to use this repository

Note that currently we only provide codes on **VGG** and **CIFAR-10** and the attack target label is **0**.<br>

### Prepare dataset
Download CIFAR-10 dataset and re-define it in the follwing format:<br>
* cifar_train['x_train'].shape = (50000, 32, 32, 3)
* cifar_test['x_test'].shape = (10000, 32, 32, 3)
* cifar_train['y_train'].shape = (50000, 1)
* cifar_test['y_test'].shape = (10000, 1)

Save the dictionary in `cifar_train` and `cifar_test` file in `./dataset` using pickle.<br>
`pickle.dump(cifar_train, open('./dataset/cifar_train', 'wb'))`<br>
`pickle.dump(cifar_test, open('./dataset/cifar_test', 'wb'))`<br>
<br>
Download Sunrise images from [Weather-Dataset](https://www.kaggle.com/rahul29g/weatherdataset) into `./CycleGAN/sunrise`.<br>

### Train your own Cycle GAN as trigger generator
Type in `python CycleGAN.py` to train your own Cycle GAN.<br>
Type in `python data_poisoning,py` to poison the training dataset.

### Perform DFST attack
Train a benign VGG as a classifer on CIFAR-10 `python train.py`.<br>
Inject the trigger using poisoned training data `python retrain.py`.<br>
Perform detoxification to force the model to learn deep features `sh run.sh`.<br>

## Contact
Free to contact the author *cheng535@purdue.edu*.
