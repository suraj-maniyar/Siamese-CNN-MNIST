from data_loader.data_loader import load_data, MyDataset
from torch.utils.data import DataLoader
from model.model import CNNModel
from trainer.trainer import train
from utils.utils import get_config_from_json

from torchsummary import summary



def main():

    # Load config
    config = get_config_from_json('config/model.config')


    # Load data
    [X_train, Y_train, X_CV, Y_CV, X_test, Y_test] = load_data(0.18)


    # Generate dataset
    train_dataset = MyDataset(X_train, Y_train)
    val_dataset = MyDataset(X_CV, Y_CV)
    test_dataset = MyDataset(X_test, Y_test)


    # Create Data Loaders
    train_loader = DataLoader(dataset = train_dataset, batch_size = config.model.batch_size, shuffle = True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = config.model.batch_size, shuffle = False)
    test_loader = DataLoader(dataset = test_dataset, batch_size = config.model.batch_size, shuffle = False)


    # Build model
    model = CNNModel()
    model = model.double()
    print(model)


    # Train model
    train(model, config, train_loader, val_loader)






if __name__ == "__main__":
    main()
