from data_loader.data_loader import load_data, MyDataset, MySiameseDataset
from torch.utils.data import DataLoader
from model.model import CNNModel, SiameseModel
from trainer.trainer import train, trainSiamese
from utils.utils import get_config_from_json

from torchsummary import summary



def mainClassification():

    # Load config
    config = get_config_from_json('config/modelClassification.config')

    # Load data
    [X_train, Y_train, X_CV, Y_CV, X_test, Y_test] = load_data(0.18)

    # Generate dataset
    train_dataset = MyDataset(X_train, Y_train)
    val_dataset = MyDataset(X_CV, Y_CV)

    # Create Data Loaders
    train_loader = DataLoader(dataset = train_dataset, batch_size = config.model.batch_size, shuffle = True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = config.model.batch_size, shuffle = False)

    # Build model
    model = CNNModel()
    model = model.double()
    print(model)


    # Train model
    train(model, config, train_loader, val_loader)


def mainSiamese():

    # Load config
    config = get_config_from_json('config/modelSiamese.config')

    # Load data
    [X_train, Y_train, X_CV, Y_CV, X_test, Y_test] = load_data(0.18)

    # Generate dataset
    train_dataset = MySiameseDataset(X_train, Y_train)
    val_dataset = MySiameseDataset(X_CV, Y_CV)

    # Create Data Loaders
    train_loader = DataLoader(dataset = train_dataset, batch_size = config.model.batch_size, shuffle = True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = config.model.batch_size, shuffle = False)

    # Build model
    model = SiameseModel()
    model = model.double()
    print(model)

    # Train model
    trainSiamese(model, config, train_loader, val_loader)



if __name__ == "__main__":
    mainClassification()
