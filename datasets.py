import os

import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms

import numpy as np

from torch.utils.data import DataLoader
import torch.utils.data as data_utils

from scipy import sparse
import urllib.request
from sklearn.preprocessing import normalize 
from sklearn.datasets import load_svmlight_file

from loss_fns import LogisticRegression, NLLSQ

from dotenv import load_dotenv
load_dotenv()


def get_dataset(name, batch_size, percentage=1.0, scale=False, loss_target_range=None):

    datasets_path = os.getenv("DATASETS_DIR")
    print(datasets_path)

    if name == "MNIST":
        assert scale == False, "Scaling not applicable."

        train_dataset, test_dataset = get_MNIST(batch_size)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False) 
        return train_loader, test_loader


    elif name == "mushrooms":

        trainX, trainY = load_svmlight_file(f"{datasets_path}/{name}")
        sample = np.random.choice(trainX.shape[0], round(trainX.shape[0] * percentage), replace=False)
        
        assert sample.shape == np.unique(sample).shape
        
        trainX = trainX[sample]
        trainY = trainY[sample]

        train_data = torch.tensor(trainX.toarray(), dtype=torch.float)
        train_target = torch.tensor(trainY, dtype=torch.float)
        
    elif name == "colon-cancer":
        trainX, trainY = load_svmlight_file(f"{datasets_path}/{name}") 

        sample = np.random.choice(trainX.shape[0], round(trainX.shape[0] * percentage), replace=False)

        assert sample.shape == np.unique(sample).shape

        trainX = trainX[sample]
        trainY = trainY[sample]


        train_data = torch.tensor(trainX.toarray(), dtype=torch.float)
        train_target = torch.tensor(trainY, dtype=torch.float)


    elif name == "covtype.libsvm.binary.scale" or name == "covtype.libsvm.binary":
        
        trainX, trainY = load_svmlight_file(f"{datasets_path}/{name}")
        sample = np.random.choice(trainX.shape[0], round(trainX.shape[0] * percentage), replace=False)

        assert sample.shape == np.unique(sample).shape

        trainX = trainX[sample]
        trainY = trainY[sample]

        train_data = torch.tensor(trainX.toarray(), dtype=torch.float)
        train_target = torch.tensor(trainY, dtype=torch.float)


    if scale:
        r1 = -10
        r2 = 10
        scaling_vec = (r1 - r2) * torch.rand(train_data.shape[1]) + r2
        scaling_vec = torch.pow(torch.e, scaling_vec)
        train_data = scaling_vec * train_data


    if loss_target_range is not None:
        train_target[train_target == train_target.unique()[0]] = loss_target_range[0]
        train_target[train_target == train_target.unique()[1]] = loss_target_range[1]

    return train_data, train_target




def get_MNIST():

    train_dataset = torchvision.datasets.MNIST(root='./datasets', 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)
                                            
    test_dataset = torchvision.datasets.MNIST(root='./datasets', 
                                            train=False, 
                                            transform=transforms.ToTensor()) 

    return train_dataset, test_dataset