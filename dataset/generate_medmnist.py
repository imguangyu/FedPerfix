import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder

# medmnist
import medmnist
from medmnist import INFO

random.seed(1)
np.random.seed(1)
num_clients = 64
dir_path = f"organamnist{num_clients}/"


# medmnist
data_flag = ("".join([i for i in dir_path if i.isalpha()])).lower()
# data_flag = 'breastmnist'
download = True
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
num_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

# Allocate data to users
def generate_dataset(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get data
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x: x.repeat(3, 1, 1)) ])

    trainset = DataClass(split='train', transform=transform, download=download)
    valset = DataClass(split='val', transform=transform, download=download)
    testset = DataClass( split='test', transform=transform, download=download)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=True)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=len(valset), shuffle=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=True)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, val_data in enumerate(valloader, 0):
        valset.data, valset.targets = val_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(valset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(valset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label).reshape(-1)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, num_classes, niid, balance, partition)
