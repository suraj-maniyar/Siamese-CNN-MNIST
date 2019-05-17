from data_loader.data_loader import load_data, MyDataset
from torch.utils.data import DataLoader
from model.model import CNNModel
from trainer.trainer import train
from utils.utils import get_config_from_json, evaluate

from torchsummary import summary
import torch
import os

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
    if os.path.isfile(config.model.path):
        model.load_state_dict(torch.load(config.model.path))
        print('Loaded checkpoint..')
    else:
        print('checkpoint not found..')

    evaluate(model, test_loader)

    #print(model)








if __name__ == "__main__":
    main()
