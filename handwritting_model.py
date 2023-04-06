import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(nn.functional.max_pool2d(x, 2))
        x = self.conv2(x)
        x = nn.functional.relu(nn.functional.max_pool2d(x, 2))
        x = x.view(-1, 1024)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the random seed for reproducibility
torch.manual_seed(0)

# Define the data pre-processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_set = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_set = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Define the data loaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

# Instantiate the model
model = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# Save the model
torch.save(model, 'handwriting_model.pth')
