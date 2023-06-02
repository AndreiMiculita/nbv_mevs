# We start with a ResNet-18 model pretrained on ImageNet,
# replace the last layer with a linear layer with 10 outputs (one for each class),
# and train the model on our dataset of rendered images.
import hashlib
import time

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection  import KFold

import os
import sys

import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import random_split
from image_dataset import ModelNet10ImageDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lr = 0.001
weight_decay = 0.0001


def main():

    # generate a unique hash to save checkpoints, based on the checksum of the starting time
    # this is useful to avoid overwriting checkpoints when running multiple experiments
    exp_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[0:8]


    # Load the pretrained ResNet-18 model
    resnet18 = torchvision.models.resnet18(pretrained=True)

    # Print the model architecture
    print(resnet18)

    # Replace the last layer with a linear layer with 10 outputs (one for each class)
    resnet18.fc = nn.Linear(512, 10)

    # Print the new model architecture
    print(resnet18)

    # Define the train transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    # Load the dataset of rendered images
    dataset = ModelNet10ImageDataset(transforms=train_transforms, train=True)

    # Split the dataset into train and validation sets
    train_ratio = 0.8
    train_size, val_size = int(train_ratio * len(dataset)), len(dataset) - int(train_ratio * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Print the size of the train and validation sets
    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')

    # Create data loaders for the train and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=2)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet18.parameters(), lr=lr, weight_decay=weight_decay)

    # Move the model to the GPU
    resnet18.to(device)

    # Train the model, with early stopping based on the validation loss
    best_val_loss = float('inf')
    best_model = None
    loss_history = []
    val_loss_history = []

    for epoch in range(50):
        print(f'Epoch {epoch}')
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_history.append(loss.item())
            if i % 100 == 99:
                print(f'Training loss for epoch {epoch}, steps {i - 99}-{i}: {running_loss / 100}')
                running_loss = 0.0
        val_loss = 0.0
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        print(f'Validation loss for epoch {epoch}: {val_loss / len(val_loader)}')
        val_loss_history.append(val_loss / len(val_loader))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = resnet18
        if len(val_loss_history) > 5 and val_loss_history[-1] > val_loss_history[-5]:
            break
    print('Finished Training')

    # Save the best model
    torch.save(best_model.state_dict(), f'{exp_hash}-resnet18_modelnet10.ckpt')

def main_kfold_cross_validation():
    """
    Same as main() but with k-fold cross validation instead
    :return:
    """
    # generate a unique hash to save checkpoints, based on the checksum of the starting time
    # this is useful to avoid overwriting checkpoints when running multiple experiments
    exp_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[0:8]

    # Load the pretrained ResNet-18 model
    resnet18 = torchvision.models.resnet18(pretrained=True)

    # Print the model architecture
    print(resnet18)

    # Replace the last layer with a linear layer with 10 outputs (one for each class)
    resnet18.fc = nn.Linear(512, 10)

    # Print the new model architecture
    print(resnet18)

    # Define the train transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    # Load the dataset of rendered images
    dataset = ModelNet10ImageDataset(transforms=train_transforms, train=True)

    # K fold cross validation
    kfold = KFold(n_splits=5, shuffle=True)


if __name__ == '__main__':
    main()
