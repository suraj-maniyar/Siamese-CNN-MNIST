import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train(model, config, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.reg)

    print(config)
    print('Training Classification Model...')

    train_loss_arr, val_loss_arr = [], []
    train_acc_arr, val_acc_arr = [], []

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

        train_loss_arr.append(train_loss.item())
        val_loss_arr.append(val_loss.item())
        train_acc_arr.append(train_accuracy)
        val_acc_arr.append(val_accuracy)
    

        print('Epoch: %2d/%2d \t Train Loss: %.4f \t Train Acc: %.3f \t Val Loss: %.3f \t Val Acc: %.3f' % (epoch+1, config.model.epochs, train_loss.item(), train_accuracy, val_loss.item(), val_accuracy))

        torch.save(model.state_dict(), 'checkpoint/modelClassification.pth')

    print('TL : ', train_loss_arr)
    print('VL : ', val_loss_arr)

    tl, = plt.plot(train_loss_arr, label='Train Loss')
    vl, = plt.plot(val_loss_arr, label='Val Loss')
    plt.legend(handles=[tl, vl])
    plt.grid()
    plt.title('Cross-Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


    print('TA : ', train_acc_arr)
    print('VA : ', val_acc_arr)

    tl, = plt.plot(train_acc_arr, label='Train Accuracy')
    vl, = plt.plot(val_acc_arr, label='Val Accuracy')
    plt.legend(handles=[tl, vl])
    plt.grid()
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()



class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin=margin

    def forward(self, output1, output2, label):
        output1 = output1.type(torch.DoubleTensor)
        output2 = output2.type(torch.DoubleTensor)
        label = label.type(torch.DoubleTensor)

        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2)+
                                        label   * torch.pow(torch.clamp(self.margin-euclidean_distance, min=0.0), 2) )

        return loss_contrastive


def trainSiamese(model, config, train_loader, val_loader):

    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.reg)

    train_loss, val_loss = [], []

    print(config)
    print('Training Siamese Model...')

    for epoch in range(0, config.model.epochs):

        model.train()
        for data in train_loader:
            img0, img1, label = data
            img0, img1, label = Variable(img0), Variable(img1), Variable(label)

            output0, output1 = model(img0, img1)

            optimizer.zero_grad()

            loss_contrastive = criterion(output0, output1, label)
            loss_contrastive.backward()
            optimizer.step()

        model.eval()
        for data in val_loader:
            img0, img1, label = data
            img0, img1, label = Variable(img0), Variable(img1), Variable(label)

            output0, output1 = model(img0, img1)

            val_loss_contrastive = criterion(output0, output1, label)

        train_loss.append(loss_contrastive.item())
        val_loss.append(val_loss_contrastive.item())
        print('Epoch: %2d/%2d   Train Loss: %.3f   Val Loss: %.3f' % (epoch, config.model.epochs, loss_contrastive.item(), val_loss_contrastive.item()))
        torch.save(model.state_dict(), 'checkpoint/modelSiamese.pth')

    print('TL : ', train_loss)
    print('VL : ', val_loss)

    tl, = plt.plot(train_loss, label='Train Loss')
    vl, = plt.plot(val_loss, label='Val Loss')
    plt.legend(handles=[tl, vl])
    plt.grid()
    plt.title('Contrastive Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
