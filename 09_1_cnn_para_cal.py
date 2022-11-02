import pydantic.schema
## Imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter    # To print to tensorboard for visualization

print(torch.__version__)

## Build Simple CNN

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5,5), stride=1, padding=0)       #kernel_size, stride, padding can be given 2 arguments or single can also work
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print("Conv1 in Layer1 : ", x.shape)
        x = self.pool1(x)
        # print("Pool1 in Layer1 : ", x.shape)
        x = F.relu(self.conv2(x))
        # print("Conv2 in Layer1 : ", x.shape)
        x = self.pool2(x)
        # print("Pool2 in Layer1 : ", x.shape)
        x = x.reshape(x.shape[0], -1)
        # print("Reshape : ", x.shape)
        x = self.fc1(x)
        # print("FC1 : ", x.shape)
        x = self.fc2(x)
        # print("FC2 : ", x.shape)
        x = self.fc3(x)
        # print("FC3 : ", x.shape)
        # m = nn.Softmax(dim=1)
        # x = m(self.fc3)
        return x

## Check Model
# model = CNN()
# x = torch.randn(4, 3, 32, 32)    # or : x = torch.randn(784, 10) : 28*28=784, MNIST greyscale images
# print('Model shape is : ', model(x).shape)           # Shapes from forward pass are printed only if model.shape is called


## Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device Being Used : ", device)

## Hyperparameters
learning_rate = 0.001
in_channels = 1
num_classes = 10
batch_size = 4
num_epochs = 2

## Load Data
train_dataset = datasets.CIFAR10(root = '/dataset', train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

## Initialize Model
model = CNN(in_channels=in_channels, num_classes=num_classes)
model.to(device)
print('Your Model Looks Like This ==> \n', model)
total_param = sum(p.numel() for p in model.parameters())
print('## Total Parameters ==> : ', total_param)

from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print('Parameter Table is ==> \n:', table)
    print(f"Total Trainable Params : {total_params}")
    return total_params

count_parameters((model))

## Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.0)

for epoch in range(num_epochs):
    losses = []
    accuracies = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)

        # Forward
        pred_scores = model(data)
        loss = criterion(pred_scores, targets)
        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Mean loss at epoch {epoch} was {sum(losses) / len(losses)}')

    #     # Calculate running training accuracy
    #     _, predictions = pred_scores.max(1)
    #     num_correct = (predictions == targets).sum()
    #
    # print(f'Mean loss at epoch {epoch} was {sum(losses)/ len(losses)}')

## Check Accuaracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on Training data")
    else:
        print("Checking accuracy on Test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x , y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()


check_accuracy(train_loader, model)
