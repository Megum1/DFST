import torch
from utils import *

from torchgeometry.losses import SSIM


class Detoxification:
    def __init__(self, config, DEVICE):
        self.detox_layers = config['detox_layers']
        self.detox_neuron_ratio = config['detox_neuron_ratio']
        self.detox_epochs = config['detox_epochs']
        self.w_ssim = config['w_ssim']
        self.w_detox = config['w_detox']

        self.alpha = config['alpha']

        self.preprocess = get_norm(config['dataset'])[0]
        self.target = config['target']
        self.device = DEVICE

        # Feature injector
        self.feat_genr = FeatureInjector().to(self.device)

    # Identify the compromised neurons
    def identify_compromised_neurons(self, model, clean_loader, poison_loader):
        # Collect the clean activation
        clean_activation = {}
        for inputs, _ in clean_loader:
            inputs = inputs.to(self.device)
            _, batch_activation = model.custom_forward(self.preprocess(inputs))
            for layer in self.detox_layers:
                if layer not in clean_activation:
                    clean_activation[layer] = []
                clean_activation[layer].append(batch_activation[layer].detach().cpu())
        
        # Collect the poisoned activation
        poison_activation = {}
        for inputs, _ in poison_loader:
            inputs = inputs.to(self.device)
            _, batch_activation = model.custom_forward(self.preprocess(inputs))
            for layer in self.detox_layers:
                if layer not in poison_activation:
                    poison_activation[layer] = []
                poison_activation[layer].append(batch_activation[layer].detach().cpu())
        
        # Identify the compromised neurons by comparing the clean and poisoned activation
        compromised_neurons = {}
        for layer in self.detox_layers:
            cl_act = torch.cat(clean_activation[layer], dim=0)
            po_act = torch.cat(poison_activation[layer], dim=0)

            # Average the values over feature maps and samples
            cl_act = cl_act.reshape(cl_act.size(0), cl_act.size(1), -1)
            cl_act = cl_act.mean(dim=2).mean(dim=0)
            po_act = po_act.reshape(po_act.size(0), po_act.size(1), -1)
            po_act = po_act.mean(dim=2).mean(dim=0)

            # Compute the difference
            diff_act = po_act - cl_act

            # Identify the compromised neurons by ranking the difference
            _, indices = torch.sort(diff_act, descending=True)
            # Take top-k neurons (k = ratio * number of neurons)
            topk = int(self.detox_neuron_ratio * diff_act.size(0))
            comp_neurons = indices[:topk].tolist()

            compromised_neurons[layer] = comp_neurons

        return compromised_neurons
    
    # Detoxify the compromised neurons by training a feature injector
    def train_feature_injector(self, model, clean_loader, compromised_neurons, verbose=False):
        # Train the feature injector
        self.feat_genr.train()
        optimizer = torch.optim.Adam(self.feat_genr.parameters(), lr=1e-2, betas=(0.5, 0.9))

        criterion_ce = torch.nn.CrossEntropyLoss()
        criterion_ssim = SSIM(window_size=5, reduction='mean')

        for epoch in range(self.detox_epochs):
            eval_loss = 0
            eval_ce_loss = 0
            eval_ssim_loss = 0
            eval_detox_loss = 0
            eval_asr = []

            for inputs, _ in clean_loader:
                inputs = inputs.to(self.device)

                optimizer.zero_grad()

                perturbs = self.feat_genr(inputs)
                detoxs = (1 - self.alpha) * inputs + self.alpha * perturbs

                with torch.no_grad():
                    outputs, activation = model.custom_forward(self.preprocess(detoxs))

                # Compute the loss
                labels = torch.full((inputs.size(0),), self.target).to(self.device)

                loss_ce = criterion_ce(outputs, labels)
                loss_ssim = criterion_ssim(detoxs, inputs)

                loss_detox = 0
                cnt_neurons = 0
                for layer in compromised_neurons:
                    for neuron in compromised_neurons[layer]:
                        loss_detox += activation[layer][:, neuron].mean()
                        cnt_neurons += 1
                if cnt_neurons > 0:
                    loss_detox /= cnt_neurons

                # Aggregate the loss
                loss = loss_ce - self.w_ssim * loss_ssim - self.w_detox * loss_detox

                # Backward pass
                loss.backward()
                optimizer.step()

                # Record the loss
                eval_loss += loss.item()
                eval_ce_loss += loss_ce.item()
                eval_ssim_loss += loss_ssim.item()
                eval_detox_loss += loss_detox.item()

                preds = outputs.max(dim=1)[1]
                eval_asr.append((preds == labels))
            
            eval_loss /= len(clean_loader)
            eval_ce_loss /= len(clean_loader)
            eval_ssim_loss /= len(clean_loader)
            eval_detox_loss /= len(clean_loader)
            eval_asr = torch.cat(eval_asr).float().mean().item()

            if (epoch + 1) % 10 == 0 and verbose:
                print(f'[Detox] Epoch {epoch+1}/{self.detox_epochs}: Loss={eval_loss:.4f}, CE={eval_ce_loss:.4f}, SSIM={eval_ssim_loss:.4f}, Detox={eval_detox_loss:.4f}, ASR={eval_asr:.4f}')
