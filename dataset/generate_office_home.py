# -*- coding: utf-8 -*-
# @Author: Jun Luo
# @Date:   2022-02-28 10:40:03
# @Last Modified by:   Jun Luo
# @Last Modified time: 2022-02-28 10:40:03

import numpy as np
from sklearn.model_selection import train_test_split

import os
import sys
import glob
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import shutil
from torch.utils.data import Dataset, DataLoader

ALPHA = 1.0
N_CLIENTS = 16
TEST_PORTION = 0.15
SEED = 1
SET_THRESHOLD = 20

IMAGE_SIZE = 224
N_CLASSES = 65

IMAGE_SRC = "./office_home/rawdata/"
SAVE_FOLDER = "./office_home16/"

class ImageDatasetFromFileNames(Dataset):
    def __init__(self, fns, labels, transform=None, target_transform=None):
        self.fns = fns
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = Image.open(self.fns[index])
        y = self.labels[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.labels)

def dirichletSplit(alpha=10, n_clients=10, n_classes=10):
    """
        alpha = infty --> balance
        alpha = 1.0 --> very imbalanced
        alpha = 0.0 --> each class is only assigned to one client
    """
    return np.random.dirichlet(n_clients * [alpha], n_classes)

def isNegligible(partitions, counts, THRESHOLD=2):
    s = np.matmul(partitions.T, counts)
    return (s < THRESHOLD).any()

def split2clientsofficehome(x_fns, ys, stats, partitions, client_idx_offset=0, verbose=False):
    print("==> splitting dataset into clients' own datasets")
    n_classes, n_clients = partitions.shape
    splits = [] # n_classes * n_clients
    for i in range(n_classes):
        indices = np.where(ys == i)[0]
        np.random.shuffle(indices)
        cuts = np.cumsum(np.round_(partitions[i] * stats[str(i)]).astype(int))
        cuts = np.clip(cuts, 0, stats[str(i)])
        cuts[-1] = stats[str(i)]
        splits.append(np.split(indices, cuts))
        
    clients = []
    for i in range(n_clients):
        indices = np.concatenate([splits[j][i] for j in range(n_classes)], axis=0)
        dset = [x_fns[indices], ys[indices]]
        clients.append(dset)
        if verbose:
            print("\tclient %03d has" % (client_idx_offset+i+1), len(dset[0]), "images")
    return clients

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

if __name__ == "__main__":
    np.random.seed(SEED)
    styles = ["Art", "Clipart", "Product", "Real World"]
    assert N_CLIENTS % 4 == 0, "### For Office-Home dataset, N_CLIENTS must be a multiple of 4...\nPlease change N_CLIENTS..."
    N_CLIENTS_PER_STYLE = N_CLIENTS // len(styles)

    cls_names = []
    for fn in get_immediate_subdirectories(IMAGE_SRC + styles[0]):
        cls_names.append(os.path.split(fn)[1])
    idx2clsname = {i: name for i, name in enumerate(cls_names)}
    get_cls_folder = lambda style, cls_n: os.path.join(IMAGE_SRC, style, cls_n)

    def get_dataset(dir, style):
        x_fns = []
        ys = []
        stats_dict = {}
        stats_list = []
        for i in range(N_CLASSES):
            cls_name = idx2clsname[i]
            x_for_cls = list(glob.glob(os.path.join(dir, style, cls_name, "*.jpg")))
            x_fns += x_for_cls
            ys += [i for _ in range(len(x_for_cls))]
            stats_dict[str(i)] = len(x_for_cls)
            stats_list.append(len(x_for_cls))
        return np.array(x_fns), np.array(ys), stats_dict, np.array(stats_list)

    clients = []
    for style_idx, style in enumerate(styles):
        dataset_style_fns, dataset_style_labels, dataset_stats_dict, dataset_stats_list = get_dataset(IMAGE_SRC, style)
        # print(len(dataset_style_fns), len(dataset_style_labels), np.sum(list(dataset_stats.values())))
        partitions = np.zeros((N_CLASSES, N_CLIENTS_PER_STYLE))
        i = 0
        while isNegligible(partitions, dataset_stats_list, SET_THRESHOLD/TEST_PORTION):
            partitions = dirichletSplit(alpha=ALPHA, n_clients=N_CLIENTS_PER_STYLE, n_classes=N_CLASSES)
            i += 1
            print(f"==> partitioning for the {i}th time (client dataset size >= {SET_THRESHOLD})")
        clients += split2clientsofficehome(dataset_style_fns,
                                           dataset_style_labels,
                                           dataset_stats_dict,
                                           partitions,
                                           client_idx_offset=style_idx*N_CLIENTS_PER_STYLE,
                                           verbose=True)
    # print()
    # print(np.sum([len(c[0]) for c in clients]))
    transform = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for client_idx, (clt_x_fns, clt_ys) in enumerate(clients):
        print("==> saving (to %s) for client [%3d/%3d]" % (SAVE_FOLDER, client_idx+1, N_CLIENTS))
        # split train, val, test
        try:
            X_train_fns, X_test_fns, y_train, y_test = train_test_split(
                clt_x_fns, clt_ys, test_size=TEST_PORTION, random_state=SEED, stratify=clt_ys)
        except ValueError:
            X_train_fns, X_test_fns, y_train, y_test = train_test_split(
                clt_x_fns, clt_ys, test_size=TEST_PORTION, random_state=SEED)

        trainset = ImageDatasetFromFileNames(X_train_fns, y_train, transform=transform)
        testset = ImageDatasetFromFileNames(X_test_fns, y_test, transform=transform)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset), shuffle=False)

        xs_train, ys_train = next(iter(trainloader))
        xs_test, ys_test = next(iter(testloader))
        train_dict = {"x": xs_train.numpy(), "y": ys_train.numpy()}
        test_dict = {"x": xs_test.numpy(), "y": ys_test.numpy()}

        # save
        for data_dict, npz_fn in [(train_dict, SAVE_FOLDER+f"train/{client_idx}.npz"), (test_dict, SAVE_FOLDER+f"test/{client_idx}.npz")]:
            with open(npz_fn, "wb") as f:
                np.savez_compressed(f, data=data_dict)
        
    print("\n==> finished saving all npz images.")
