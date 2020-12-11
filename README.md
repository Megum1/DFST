DFST
===

This is the repository for DFST paper *Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification*.<br>

Dependences
---

Python3.6, tensorflow=1.13.1, keras=2.2.4, numpy, pickle, PIL.<br>

How to use this repository
---

Note that currently we only provide codes on *VGG* and *CIFAR-10* and the attack target label is *0*.<br>

###Prepare dataset
Download CIFAR-10 dataset and re-define it in the follwing format:<br>
* dictionary['x_train'].shape = (50000, 32, 32, 3)
* dictionary['x_test'].shape = (10000, 32, 32, 3)
* dictionary['y_train'].shape = (50000, 1)
* dictionary['y_test'].shape = (10000, 1)
Save the dictionary in `cifar_train` and `cifar_test` file in `./dataset` using pickle.<br>
		pickle.dump(dictionary, open('./dataset/***', 'wb'))
<br>
Download Sunrise images from [Weather-Dataset](https://www.kaggle.com/rahul29g/weatherdataset) into `./CycleGAN/sunrise`.<br>

###Train your own Cycle GAN as trigger generator
Type in
		python CycleGAN.py
to train your own Cycle GAN.<br>
Type in 
		data_poisoning,py
to poison the training dataset.

###Perform DFST attack
Train a benign VGG as a classifer on CIFAR-10.
		python train.py
Inject trigger using poisoned training data.
		python retrain.py
Perform detoxification to force the model to learn deep features.
		sh run.sh

Contact
---
Free to contact the author *cheng535@purdue.edu*
