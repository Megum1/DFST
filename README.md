# DFST
train.py is used to train benign models;
retrain.py is used to retrain the pretrained models with style-transferred images;
CycleGAN folder is the Cycle GAN implementation, and you can run CycleGAN.py to train the model while make_trojan.py use the trained generator to make the new dataset for data-poisoning;
detoxification folder is the key step of the project, and you can run run.sh to detoxificate the trojaned model and escape from the simple-tuned ABS detection;

Here the parameters are tailored for CIFAR-10. For more experiments, you need to modify the preprocessing step in each .py file;

https://jbox.sjtu.edu.cn/l/qJlyCg is the link of the dataset.zip;
In dataset folder, cifar and sunrise are used for Cycle GAN training, while cifar_train and cifar_test is used for classifier training.
