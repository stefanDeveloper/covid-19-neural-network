import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

from binary_classifier.model import Net
from dataset import Dataset


def train_model(images, labels, epochs=10, learning_rate=0.0001, batch_size=32):
    net = Net()

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Generate dataset
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
            loss = criterion(outputs, labels)

            # Accuracy
            predicted = torch.round(outputs.data).reshape(len(labels))
            total = labels.size(0)
            correct = (predicted == labels).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {} %'
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
            predicted = torch.round(outputs.data).reshape(len(labels))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

    # Save the model checkpoint
    torch.save(net.state_dict(), './models/model_simpleCNN_bin_covid.ckpt')
