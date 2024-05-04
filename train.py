# coding: utf-8

import warnings
from xml.dom import xmlbuilder
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
import time
import copy
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from attack import *
from utils import *
from detoxification import *


def eval_acc(model, loader, preprocess, DEVICE):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = model(preprocess(x_batch))
            pred = output.max(dim=1)[1]

            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc


def train(config, save_folder, logger, DEVICE):
    # Set random seed
    seed_torch(config['seed'])

    # Load model
    model = get_model(config['dataset'], config['network']).to(DEVICE)

    # Load dataset
    train_set = get_dataset(config['dataset'], train=True)
    test_set  = get_dataset(config['dataset'], train=False)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    preprocess, _ = get_norm(config['dataset'])

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    time_start = time.time()
    for epoch in range(config['epochs']):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(preprocess(x_batch))
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        time_end = time.time()
        acc = eval_acc(model, test_loader, preprocess, DEVICE)

        # Log the training process
        logger.info(f'epoch {epoch} - {time_end-time_start:.2f}s, acc: {acc:.4f}')
        time_start = time.time()

        scheduler.step()

        save_path = f'{save_folder}/model.pt'
        torch.save(model, save_path)


def test(config, save_folder, DEVICE):
    # Set random seed
    seed_torch(config['seed'])

    model_filepath = f'{save_folder}/model.pt'
    model = torch.load(model_filepath, map_location='cpu').to(DEVICE)
    model.eval()

    preprocess, _ = get_norm(config['dataset'])

    test_set = get_dataset(config['dataset'], train=False)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=config['batch_size'])

    acc = eval_acc(model, test_loader, preprocess, DEVICE)

    if config['attack'] == 'clean':
        print(f'Accuarcy: {acc*100:.2f}%')
    else:
        # Load poisoned dataset
        poison_data = torch.load(os.path.join(save_folder, 'poison_data.pt'))
        poison_x_test = poison_data['test']
        poison_y_test = torch.full((poison_x_test.size(0),), config['target'])

        poison_set = CustomDataset(poison_x_test, poison_y_test)
        poison_loader = DataLoader(poison_set, batch_size=config['batch_size'])

        asr = eval_acc(model, poison_loader, preprocess, DEVICE)
        print(f'Accuarcy: {acc*100:.2f}%, ASR: {asr*100:.2f}%')


def poison(config, save_folder, logger, DEVICE):
    # Set random seed
    seed_torch(config['seed'])

    # Load model
    model = get_model(config['dataset'], config['network']).to(DEVICE)

    # Initialize backdoor
    backdoor = get_backdoor(config, DEVICE)

    # Create poisoned dataset
    attack = Attack(config, backdoor, save_folder)

    # Load dataset
    train_loader  = DataLoader(dataset=attack.train_set,
                               batch_size=config['batch_size'],
                               shuffle=True)
    poison_loader = DataLoader(dataset=attack.poison_test_set,
                               batch_size=config['batch_size'])
    test_loader   = DataLoader(dataset=attack.test_set,
                               batch_size=config['batch_size'])
    
    # Preprocess (Normalization)
    preprocess, _ = get_norm(config['dataset'])

    # Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Training
    best_acc = 0
    best_asr = 0
    time_start = time.time()
    for epoch in range(config['epochs']):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = attack.inject(x_batch, y_batch)

            optimizer.zero_grad()

            output = model(preprocess(x_batch))
            loss = criterion(output, y_batch)

            loss.backward()
            optimizer.step()

            pred = output.max(dim=1)[1]
            acc = (pred == y_batch).sum().item() / x_batch.size(0)

            if step % 10 == 0:
                sys.stdout.write('\repoch {:3}, step: {:4}, loss: {:.4f}, '
                                 .format(epoch, step, loss) +\
                                 'acc: {:.4f}'.format(acc))
                sys.stdout.flush()

        scheduler.step()

        time_end = time.time()
        acc = eval_acc(model, test_loader, preprocess, DEVICE)
        asr = eval_acc(model, poison_loader, preprocess, DEVICE)

        # Log the training process
        logger.info(f'epoch {epoch} - {time_end-time_start:.2f}s, acc: {acc:.4f}, asr: {asr:.4f}')
        time_start = time.time()

        # Save the model if the performance is better
        if acc + asr > best_acc + best_asr:
            best_acc = acc
            best_asr = asr
            logger.info(f'---BEST ACC: {best_acc:.4f}, ASR: {best_asr:.4f}---')
            torch.save(model, f'{save_folder}/model.pt')
        
        ######################################################
        # (DFST) Apply detoxification in the middle of training
        ######################################################
        if config['attack'] == 'dfst' and config['detox_flag']:
            # Set the model to evaluation mode
            model.eval()

            # Initialize the detoxification module
            detox = Detoxification(config, DEVICE)

            # Collect 3% of the clean/poison data for detoxification
            detox_clean_x = []
            detox_clean_y = []
            for x_clean, y_clean in train_loader:
                detox_clean_x.append(x_clean)
                detox_clean_y.append(y_clean)
                if len(detox_clean_x) == int(len(train_loader) * 0.03):
                    break
            detox_clean_x = torch.cat(detox_clean_x)
            detox_clean_y = torch.cat(detox_clean_y)
            detox_clean = CustomDataset(detox_clean_x, detox_clean_y)
            detox_clean_loader = DataLoader(detox_clean, batch_size=config['batch_size'], shuffle=True)

            detox_poison_x = []
            detox_poison_y = []
            for x_poison in attack.poison_x_train:
                detox_poison_x.append(x_poison)
                detox_poison_y.append(config['target'])
                if len(detox_poison_x) == len(detox_clean):
                    break
            detox_poison_x = torch.stack(detox_poison_x)
            detox_poison_y = torch.tensor(detox_poison_y)
            detox_poison = CustomDataset(detox_poison_x, detox_poison_y)
            detox_poison_loader = DataLoader(detox_poison, batch_size=config['batch_size'], shuffle=True)

            # Get the compromised neurons
            compromised_neurons = detox.identify_compromised_neurons(model, detox_clean_loader, detox_poison_loader)

            # Train the feature injector
            detox.train_feature_injector(model, detox_clean_loader, compromised_neurons, verbose=True)

            # Apply the feature injector
            attack.feat_genr = detox.feat_genr
            attack.feat_genr.eval()
