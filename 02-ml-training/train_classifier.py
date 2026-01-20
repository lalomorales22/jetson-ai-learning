#!/usr/bin/env python3
"""
Train a CNN image classifier on CIFAR-10
Demonstrates GPU-accelerated training on Jetson
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_WORKERS = 4

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification"""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 32x32 -> 16x16

        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 16x16 -> 8x8

        # Conv block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # 8x8 -> 4x4

        # Flatten
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def train_epoch(model, dataloader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for i, (images, labels) in enumerate(dataloader):
        # Move to GPU
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'  Batch [{i+1}/{len(dataloader)}], '
                  f'Loss: {running_loss/(i+1):.3f}, '
                  f'Acc: {100.*correct/total:.2f}%')

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    print(f'Epoch {epoch} completed in {epoch_time:.2f}s - '
          f'Loss: {epoch_loss:.3f}, Acc: {epoch_acc:.2f}%')

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    print(f'Validation - Loss: {val_loss:.3f}, Acc: {val_acc:.2f}%')

    return val_loss, val_acc


def main():
    print("=" * 60)
    print("CIFAR-10 Image Classification Training")
    print("=" * 60)

    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load datasets
    print("\nLoading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")

    # Create model
    print("\nInitializing model...")
    model = SimpleCNN().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print("=" * 60)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, testloader, criterion)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"✓ New best model saved! (Acc: {val_acc:.2f}%)")

    print("\n" + "=" * 60)
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    print("=" * 60)

    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("\nModels saved:")
    print("  - best_model.pth (best validation accuracy)")
    print("  - final_model.pth (final epoch)")


if __name__ == '__main__':
    main()
