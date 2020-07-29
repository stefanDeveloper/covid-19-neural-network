import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

LEARNING_RATE = 0.001

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128 * 26 * 26, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
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
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    train_data = []
    for i in range(len(images)):
        train_data.append([images[i], labels[i]])

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            #images = images.reshape(-1, 224 * 224)
            images = images.reshape(len(images), 1, 224, 224)
            labels = labels

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            # images = images.reshape(-1, 224 * 224)
            labels = labels
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(net.state_dict(), 'model.ckpt')
