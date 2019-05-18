from data_loader.data_loader import load_data, MyDataset, MySiameseDataset
from torch.utils.data import DataLoader
from model.model import CNNModel, SiameseModel
from utils.utils import get_config_from_json, evaluate

import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import numpy as np

from matplotlib import rcParams
rcParams['axes.titlepad'] = 1

def main():

    # Load config
    config = get_config_from_json('config/model.config')

    # Load data
    [X_train, Y_train, X_CV, Y_CV, X_test, Y_test] = load_data(0.18)

    # Generate dataset
    test_dataset = MyDataset(X_test, Y_test)

    # Create Data Loaders
    test_loader = DataLoader(dataset = test_dataset, batch_size = config.model.batch_size, shuffle = False)


    # Build model
    model = CNNModel()
    model = model.double()
    model.eval()

    if os.path.isfile(config.model.path):
        model.load_state_dict(torch.load(config.model.path))
        print('Loaded checkpoint..')
    else:
        print('checkpoint not found..')

    evaluate(model, test_loader)



def mainSiamese():
    # Load config
    config = get_config_from_json('config/modelSiamese.config')

    # Load data
    [X_train, Y_train, X_CV, Y_CV, X_test, Y_test] = load_data(0.18)

    # Generate dataset
    test_dataset = MySiameseDataset(X_test, Y_test)

    # Create Data Loaders
    test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = True)


    # Build model
    model = SiameseModel()
    model = model.double()
    print(model)
    model.eval()

    if os.path.isfile(config.model.path):
        model.load_state_dict(torch.load(config.model.path))
        print('Loaded checkpoint..')
    else:
        print('checkpoint not found..')

    dataiter = iter(test_loader)
    x0, _, _ = next(dataiter)

    plt.tight_layout()

    plt.subplot(4,3,2)
    plt.axis('off')
    plt.imshow(x0[0][0], 'gray')
    plt.title('Original Image', fontdict = {'fontsize' : 10})


    for i in range(9):
        _, x1, label = next(dataiter)
        output0, output1 = model(x0, x1)

        output0 = output0.type(torch.DoubleTensor)
        output1 = output1.type(torch.DoubleTensor)

        euclidean_distance = F.pairwise_distance(output0, output1)

        plt.subplot(4, 3, i+4)
        plt.axis('off')
        plt.imshow(x1[0][0], 'gray')
        plt.title( str(round( euclidean_distance.item(), 2) ), fontdict = {'fontsize' : 10})

    plt.show()

if __name__ == "__main__":
    mainSiamese()
