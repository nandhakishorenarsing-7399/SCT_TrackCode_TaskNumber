import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Generate sample dataset if not exists
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(os.path.join(dataset_dir, 'cats'))
    os.makedirs(os.path.join(dataset_dir, 'dogs'))

    # Generate 20 images for cats (red squares)
    for i in range(20):
        img = Image.new('RGB', (64, 64), color=(255, 0, 0))  # Red
        img.save(os.path.join(dataset_dir, 'cats', f'cat_{i}.jpg'))

    # Generate 20 images for dogs (blue squares)
    for i in range(20):
        img = Image.new('RGB', (64, 64), color=(0, 0, 255))  # Blue
        img.save(os.path.join(dataset_dir, 'dogs', f'dog_{i}.jpg'))

# Dataset path
dataset_path = 'dataset'

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = len(train_dataset.classes)
model = CNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train
for epoch in range(5):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

# Save model
torch.save(model.state_dict(), "gesture_model.pth")
print("Model saved as gesture_model.pth")