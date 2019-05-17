import numpy as np
import matplotlib.pyplot as plt
import struct
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def load_data(split=0.18):
    X_train = read_idx('../data/train-images.idx3-ubyte')
    Y_train = read_idx('../data/train-labels.idx1-ubyte')

    X_test = read_idx('../data/t10k-images.idx3-ubyte')
    Y_test = read_idx('../data/t10k-labels.idx1-ubyte')

    X_train, X_CV, Y_train, Y_CV = train_test_split(X_train, Y_train, test_size=split, random_state=42)

    X_train = np.expand_dims(X_train, 1).astype('float64')
    X_CV = np.expand_dims(X_CV, 1).astype('float64')
    X_test = np.expand_dims(X_test, 1).astype('float64')

    Y_train = Y_train.astype('int64')
    Y_CV = Y_CV.astype('int64')
    Y_test = Y_test.astype('int64')


    print('Train : ')
    print(X_train.shape)
    print(Y_train.shape)
    print('CV : ')
    print(X_CV.shape)
    print(Y_CV.shape)
    print('Test : ')
    print(X_test.shape)
    print(Y_test.shape)


    return [X_train, Y_train, X_CV, Y_CV, X_test, Y_test]


class MyDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])
