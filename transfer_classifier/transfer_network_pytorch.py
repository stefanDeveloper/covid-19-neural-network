import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

LEARNING_RATE = 0.001


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)

        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        # x = x.view(-1, self.num_flat_features(x))
        x = x.view(-1, 128 * 26 * 26)

        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_model(images, labels, epochs=10):
    net = Net()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    data = []
    for i in range(len(images)):
        data.append([images[i], labels[i]])

    train_data, test_data = train_test_split(data, test_size=0.2)

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(len(images), 1, 224, 224)
            labels = labels

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, torch.max(labels, 1)[1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

    # Test the model
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=100)

    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(len(images), 1, 224, 224)
            labels = labels
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(net.state_dict(), './models/model_multiCNN_bin_covid.ckpt')


def train_using_pretrained_model(images, labels, model, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    data = []
    for i in range(len(images)):
        data.append([images[i], labels[i]])

    train_data, test_data = train_test_split(data, test_size=0.2)

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)

    for images, labels in iter(train_loader):
        images = images
        labels = labels
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
