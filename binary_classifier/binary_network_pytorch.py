import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

from binary_classifier.model import Net
from dataset import Dataset
from utils import plot_loss, plot_acc, plot_roc_binary


def train_model(images, labels, path, epochs=10, learning_rate=0.0001, batch_size=32):
    net = Net()
    train_loss, test_loss,  = [], []
    train_acc, test_acc,  = [], []
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
        train_loss_it, train_acc_it = [], []
        test_loss_it, test_acc_it = [], []
        roc_score_it, roc_true_it = [], []
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

            train_acc_it.append(accuracy)
            train_loss_it.append(loss.item())

        train_acc.append(np.mean(np.array(train_acc_it)))
        train_loss.append(np.mean(np.array(train_loss_it)))

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

                roc_score_it.extend(np.array(outputs.data).reshape(-1))
                roc_true_it.extend(np.array(labels).reshape(-1))

                test_acc_it.append(accuracy)
                test_loss_it.append(loss.item())

                print('[Test] Epoch [{}/{}], Step [{}/{}], Test-Loss: {:.4f}, Test-Acc: {:.2f}%'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item(), accuracy))

            if (epoch + 1) % 10 == 0:
                roc_score.append(roc_score_it)
                roc_true.append(roc_true_it)

            test_acc.append(np.mean(np.array(test_acc_it)))
            test_loss.append(np.mean(np.array(test_loss_it)))

    # Save the model checkpoint
    torch.save(net.state_dict(), path)

    # ROC
    if epochs > 9:
        true = np.array(roc_true)
        score = np.array(roc_score)
        print(true.shape)


        plot_roc_binary(true, score, './results/simple_classifier_roc.pdf', 'Simple Binary Classifier')
    plot_loss(train_loss, test_loss, './results/simple_classifier_loss.pdf', 'Simple Binary Classifier')
    plot_acc(train_acc, test_acc, './results/simple_classifier_acc.pdf', 'Simple Binary Classifier')

    return net
