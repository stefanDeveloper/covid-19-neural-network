import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from binary_classifier.model import Net
from dataset import Dataset
from utils import plot_loss, plot_acc
import matplotlib.pyplot as plt


def train_model(images, labels, path, epochs=10, learning_rate=0.0001, batch_size=32):
    net = Net()
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    roc_score = []
    roc_true = []

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Generate dataset
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=6)
    train_data = Dataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

    test_data = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):
        net.train()
        total_step = len(train_loader)
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(len(images), 1, 224, 224)
            labels = labels

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Accuracy
            predicted = torch.round(outputs.data).reshape(len(labels))
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[Train] Epoch [{}/{}], Step [{}/{}], Train-Loss: {:.4f}, Train-Acc: {:.2f} %'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item(), accuracy))

            if i % total_step == 0:
                train_acc.append(accuracy)
                train_loss.append(loss)

        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_step = len(test_loader)
            for i, (images, labels) in enumerate(test_loader):
                images = images.reshape(len(images), 1, 224, 224)
                labels = labels
                outputs = net(images)

                loss = criterion(outputs, labels)

                predicted = torch.round(outputs.data).reshape(len(labels))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total

                roc_score.append(predicted)
                roc_true.append(labels)

                if i % total_step == 0:
                    test_acc.append(accuracy)
                    test_loss.append(loss)

                print('[Test] Epoch [{}/{}], Step [{}/{}], Test-Loss: {:.4f}, Test-Acc: {:.2f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item(), accuracy))

    # Save the model checkpoint
    torch.save(net.state_dict(), path)

    # ROC
    fpr, tpr, thresholds = metrics.roc_curve(np.array(roc_true).reshape(-1), np.array(roc_score).reshape(-1))

    plot_roc_bin(fpr, tpr, './results/simple_classifier_roc.pdf')
    plot_loss(train_loss, test_loss, './results/simple_classifier_loss.pdf')
    plot_acc(train_acc, test_acc, './results/simple_classifier_acc.pdf')

    return net
