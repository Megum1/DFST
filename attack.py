from tqdm import tqdm
from torchvision.utils import save_image

from backdoors import *
from utils import *


# The Attack class is responsible for constructing the poisoned dataset
class Attack:
    def __init__(self, config, backdoor, save_folder):
        # Extract the configuration parameters
        self.dataset = config['dataset']
        self.attack = config['attack']
        self.target = config['target']
        self.poison_rate = config['poison_rate']
        self.batch_size = config['batch_size']

        self.backdoor = backdoor
        self.save_folder = save_folder

        # Device
        self.device = backdoor.device

        # (DFST) Feature injector
        if self.attack == 'dfst':
            self.alpha = config['alpha']
            self.feat_genr = None

        # Initialize the dataset (without augmentation)
        self.train_set = get_dataset(self.dataset, train=True, augment=False)
        self.test_set = get_dataset(self.dataset, train=False, augment=False)

        # Augmentation
        shape = self.train_set[0][0].shape[-2:]
        self.augment = PostTensorTransform(shape)

        # Construct the poisoned dataset and save in the save_folder
        # Poisoned data are current saved in .pt format for faster loading
        # Can be saved in individual images if the memory is not sufficient
        if os.path.exists(os.path.join(self.save_folder, 'poison_data.pt')):
            print('Loading the poisoned data...')
            poison_data = torch.load(os.path.join(self.save_folder, 'poison_data.pt'))
        else:
            print('Constructing the poisoned training set...')
            poison_x_train = self.stamp_trigger(self.train_set)

            print('Constructing the poisoned test set...')
            poison_x_test = self.stamp_trigger(self.test_set)

            # Save the poisoned data
            poison_data = {'train': poison_x_train, 'test': poison_x_test}
            torch.save(poison_data, os.path.join(self.save_folder, 'poison_data.pt'))
            print('Poisoned data saved in:', os.path.join(self.save_folder, 'poison_data.pt'))

            # Sample 10 poisoned images as illustration
            savefig = poison_x_train[:10]
            save_image(savefig, os.path.join(self.save_folder, 'visual_poison.png'), nrow=10)

        # Poisoned training set
        self.poison_x_train = poison_data['train']

        # Poisoned test set
        poison_y_test = torch.full((poison_data['test'].size(0),), self.target)
        self.poison_test_set = CustomDataset(poison_data['test'], poison_y_test)
    
    # Stamp the trigger on the non-target samples
    def stamp_trigger(self, dataset):
        non_target = []
        for image, label in dataset:
            if label != self.target:
                non_target.append(image)
        non_target = torch.stack(non_target)
        num_batches = int(np.ceil(len(non_target) / self.batch_size))
        x_poison = []
        for i in tqdm(range(num_batches)):
            x_batch = non_target[i * self.batch_size:(i + 1) * self.batch_size].to(self.device)
            x_batch_poison = self.backdoor.inject(x_batch)
            x_poison.append(x_batch_poison.cpu())
        x_poison = torch.cat(x_poison)
        return x_poison

    # Mix the poisoned data into the training set
    def inject(self, inputs, labels):
        # Number of poisoned samples within the batch
        num_bd = int(inputs.size(0) * self.poison_rate)

        # If the batch size is too small, we may not have any poisoned samples
        if num_bd == 0:
            return self.augment(inputs), labels

        # Randomly sample num_bd samples from the poisoned dataset
        indices = torch.randperm(self.poison_x_train.size(0))[:num_bd]
        inputs_bd = self.poison_x_train[indices].to(self.device)
        labels_bd = torch.full((num_bd,), self.target).to(self.device)

        if self.attack == 'dfst':
            # Number of detox samples within the batch
            num_dx = int(inputs.size(0) * self.poison_rate)

            # Take the detox samples from the poisoned dataset
            inputs_dx = inputs[num_bd:num_bd + num_dx].to(self.device)
            labels_dx = labels[num_bd:num_bd + num_dx].to(self.device)

            # Generate the detox samples if the feature injector is available
            if self.feat_genr is not None:
                perturbs = self.feat_genr(inputs_dx)
                inputs_dx = (1 - self.alpha) * inputs_dx + self.alpha * perturbs

            # Combine the poisoned and detox samples
            inputs_bd = torch.cat((inputs_bd, inputs_dx), dim=0)
            labels_bd = torch.cat((labels_bd, labels_dx), dim=0)

            # The remaining samples are clean
            inputs_cl = inputs[num_bd + num_dx:].to(self.device)
            labels_cl = labels[num_bd + num_dx:].to(self.device)

        else:
            # The remaining samples are clean
            inputs_cl = inputs[num_bd:].to(self.device)
            labels_cl = labels[num_bd:].to(self.device)

        # Mix the poisoned and clean samples
        inputs = torch.cat((inputs_bd, inputs_cl), dim=0)
        labels = torch.cat((labels_bd, labels_cl), dim=0)

        # Augment the batch
        inputs = self.augment(inputs)

        return inputs, labels
