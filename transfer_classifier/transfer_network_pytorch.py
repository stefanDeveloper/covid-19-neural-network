import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

from dataset import Dataset
from transfer_classifier.model import Net


def train_model(images, labels, epochs=10, learning_rate=0.0001, batch_size=32):
    net = Net()

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=6)
    train_data = Dataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(epochs):
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

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item(), 100 * correct / total))

    # Test the model
    test_data = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.reshape(len(images), 1, 224, 224)
            labels = labels
            outputs = net(images)
            predicted = torch.round(outputs.data)
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

    # Save the model checkpoint
    torch.save(net.state_dict(), './models/model_multiCNN_bin_covid.ckpt')


def train_using_pretrained_model(images, labels, net, epochs=10, learning_rate=0.0001, batch_size=32):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=6)
    train_data = Dataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

    for images, labels in iter(train_loader):
        images = images
        labels = labels
        optimizer.zero_grad()
        outputs = net(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

    # Save the model checkpoint
    torch.save(net.state_dict(), './models/model_multiCNN_bin_covid.ckpt')
