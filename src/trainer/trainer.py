import torch
import torch.nn as nn
from torch.autograd import Variable

def train(model, config, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.reg)

    print(config)
    print('Training...')

    for epoch in range(config.model.epochs):
        total, correct = 0, 0
        for images, labels in train_loader:
            images = Variable(images)
            labels = Variable(labels)
            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum(0)
            train_loss = criterion(outputs, labels)

            train_loss.backward()
            optimizer.step()
        train_accuracy = 100.0 * correct/total

        total, correct = 0, 0
        for images, labels in val_loader:
            images = Variable(images)
            labels = Variable(labels)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum(0)
            val_loss = criterion(outputs, labels)
            
        val_accuracy = 100.0 * correct/total

        print('Epoch: %2d/%2d \t Train Loss: %.4f \t Train Acc: %.3f \t Val Loss: %.3f \t Val Acc: %.3f' % (epoch+1, config.model.epochs, train_loss.item(), train_accuracy, val_loss.item(), val_accuracy))
        torch.save(model.state_dict(), 'model.pth')
