from dotmap import DotMap
import json
import torch.nn as nn
from torch.autograd import Variable
import torch


def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = DotMap(config_dict)
    return config

def evaluate(model, test_loader):

    criterion = nn.CrossEntropyLoss()

    total, correct = 0, 0
    y_pred, y_true = [], []
    for images, labels in test_loader:
        images = Variable(images)
        labels = Variable(labels)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        y_true += labels.tolist()
        y_pred += predicted.tolist()

        total += labels.size(0)
        correct += (labels == predicted).sum(0)
        
        test_loss = criterion(outputs, labels)

    test_accuracy = 100.0 * correct.item()/total

    print('Accuracy : ', test_accuracy)
