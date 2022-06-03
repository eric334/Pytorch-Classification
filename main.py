import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

train_data = datasets.CIFAR10(
    root = 'data',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

test_data = datasets.CIFAR10(
    root = 'data',
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

image_type = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(train_data)
print(test_data)

def imshow(img):
    plt.imshow(np.transpose((img).numpy(), (1, 2, 0)))

def create_converted_subplot_figure(images, titles, size, format):
    fig = plt.figure(figsize=size)
    for i, (image, title) in enumerate(zip(images, titles)):
        fig.add_subplot(format[0], format[1], i+1)
        imshow(image) 
        plt.axis('off')
        plt.title(str(title))

def create_data_subplot_figure(data_list, titles, size, format):
    fig = plt.figure(figsize=size)
    for i, (data, title) in enumerate(zip(data_list, titles)):
        fig.add_subplot(format[0], format[1], i+1)
        plt.plot(data)
        plt.title(str(title))

samples = []
for i in range(10):
    j = 0
    while test_data[j][1] != i:
        j += 1
    samples.append(test_data[j])

create_converted_subplot_figure(
    [sample[0] for sample in samples],
    ["%s"%image_type[sample[1]] for sample in samples],
    (5,10),
    [5,2])
plt.savefig("visualize10.jpg")
plt.show()

train_loader = DataLoader(dataset=train_data, batch_size=15, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=15, shuffle=True)

class HW7Net(nn.Module):
  def __init__(self) -> None:
      super(HW7Net, self).__init__()

      self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,stride=1) # 
      self.pool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,stride=1)
      self.pool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.dense_layer1 = nn.Linear(in_features=64 * 5 * 5, out_features=10)
      self.dropout_layer1 = nn.Dropout(p=0.4)
      self.dense_layer2 = nn.Linear(in_features=10, out_features=10)

  def forward(self, x):
    x = F.relu(self.conv_layer1(x))
    x = self.pool_layer1(x)
    x = F.relu(self.conv_layer2(x))
    x = self.pool_layer2(x)
    x = torch.flatten(x, 1)
    x = self.dense_layer1(x)
    x = self.dropout_layer1(x)
    x = self.dense_layer2(x)

    return x

net = HW7Net()
print(net)
print(f"Total Parameters: {sum(p.numel() for p in net.parameters())}")
print(f"Trainable Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr = .03)

net.to(device)

num_epochs = 60

def train_network():
    global net
    global optimizer
    global train_loader
    global loss_function
    global device

    net.train()
    correct = 0
    total = 0
    loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        total += labels.size(0)
        predicted = torch.max(outputs.data, 1)[1]
        correct += (predicted == labels).sum().item()

        #print(f"predicted: {predicted.shape}")
        #print(f"correct: {correct}")

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    return correct/total, loss.item()

def test_network():
    global net
    global test_loader
    global loss_function
    global device
    net.eval()
    loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = loss_function(outputs, labels)
            predicted = torch.max(outputs.data, 1)[1]

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total, loss.item()

def get_incorrect_images():
    global test_data
    global net
    global image_type
    net.eval()
    losses_images = [None] * 10 
    losses_titles = [None] * 10 
    added = 0

    temp_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    for i, (images, labels) in enumerate(temp_loader):
        images = images.to(device)
        item = labels.to(device).item()
        #print (image_type[item])
        predicted = torch.max(net(images).data, 1)[1].item()
        if predicted != item and losses_images[item] is None:
            added += 1
            losses_images[item] = test_data[i][0]
            losses_titles[item] = f"Actual: {image_type[item]}  Guess: {image_type[predicted]}"

        if added == 10:
            break

    return losses_images, losses_titles

train_accuracies = []
train_losses = []

test_accuracies = []
test_losses = []

for epoch in range(num_epochs):
    print(f'Epoch: {epoch+1}/{num_epochs}')

    accuracy, loss = train_network()
    print(f'    Train Accuracy: {accuracy*100} %  Train Loss: {loss}')
    train_accuracies.append(accuracy)
    train_losses.append(loss) 

    accuracy, loss = test_network()
    print(f'    Test Accuracy: {accuracy*100} %  Test Loss: {loss}')
    test_accuracies.append(accuracy)
    test_losses.append(loss) 

losses_images, losses_titles = get_incorrect_images()

# incorrectly classified images
create_converted_subplot_figure(
    losses_images,
    losses_titles,
    (8,10),
    [5,2])
plt.savefig("misclassified.jpg")
plt.show()

# data plot for training
create_data_subplot_figure(
    [   
        train_accuracies,
        train_losses,
        test_accuracies,
        test_losses
    ],
    [
        "Train Accuracy",
        "Train Loss",
        "Test Accuracy",
        "Test Loss"
    ],
    (8,8),
    [2,2])
plt.savefig("training_data_plots.jpg")
plt.show()