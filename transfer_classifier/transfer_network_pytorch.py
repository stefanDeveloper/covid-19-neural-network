import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

from dataset import Dataset
from transfer_classifier.model import Net
import matplotlib.pyplot as plt

from utils import plot_loss, plot_acc


def train_model(images, labels, path, epochs=10, learning_rate=0.0001, batch_size=32):
    net = Net()
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=6)
    train_data = Dataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

    test_data = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(epochs):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(len(images), 1, 224, 224)
            labels = labels

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs.double(), labels)

            # Accuracy
            predicted = torch.round(outputs.data)
            total = labels.size(0) * labels.size(1)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[Train] Epoch [{}/{}], Step [{}/{}], Train-Loss: {:.4f}, Train-Acc: {:.2f}'
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

                loss = criterion(outputs.double(), labels)

                predicted = torch.round(outputs.data)
                total += labels.size(0) * labels.size(1)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total

                if i % total_step == 0:
                    test_acc.append(accuracy)
                    test_loss.append(loss)

                print('[Test] Epoch [{}/{}], Step [{}/{}], Train-Loss: {:.4f}, Train-Acc: {:.2f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item(), accuracy))

    # Save the model checkpoint
    torch.save(net.state_dict(), path)

    plot_loss(train_loss, test_loss, './results/multiple_classifier_loss.pdf')
    plot_acc(train_acc, test_acc, './results/multiple_classifier_acc.pdf')

    return net


def train_using_pretrained_model(images, labels, path, net, epochs=10, learning_rate=0.0001, batch_size=32):
    best_accuracy = 0.0
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Training data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=6)
    train_data = Dataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

    # Testing data
    test_data = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(len(images), 1, 224, 224)
            labels = labels
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs.double(), labels)

            loss.backward()
            optimizer.step()

            # Accuracy
            predicted = torch.round(outputs.data)
            total = labels.size(0) * labels.size(1)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            print('Epoch [{}/{}], Step [{}/{}], Train-Loss: {:.4f}, Train-Acc: {:.2f} %'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item(), accuracy))

            if i % total_step == 0:
                train_acc.append(accuracy)
                train_loss.append(loss)

        total = 0.0
        correct = 0.0
        total_step = len(test_loader)
        for i, (images, labels) in enumerate(test_loader):
            images = images.reshape(len(images), 1, 224, 224)
            labels = labels
            outputs = net(images)

            loss = criterion(outputs.double(), labels)

            predicted = torch.round(outputs.data)
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            if i % total_step == 0:
                test_acc.append(accuracy)
                test_loss.append(loss)

        test_accuracy = 100 * correct / total

        print('[Test] Epoch [{}/{}], Acc: {:.2f}'.format(epoch + 1, epochs, test_accuracy))

        if test_accuracy > best_accuracy:
            torch.save(net.state_dict(), path)
            best_accuracy = test_accuracy

    plot_loss(train_loss, test_loss, './results/transfer_classifier_loss.pdf')
    plot_acc(train_acc, test_acc, './results/transfer_classifier_acc.pdf')
