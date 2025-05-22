import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, F1Score, ConfusionMatrix
import seaborn as sns
from tqdm.auto import tqdm # progress bar

'''
This is a simple implementation of a Convolutional Neural Network (CNN) for a multi-class image classification.
The images come from the FashionMNIST Dataset.
The code is mostly based on this implementation: https://www.learnpytorch.io/03_pytorch_computer_vision/
'''

from timeit import default_timer as timer 
def print_train_time(start: float, end: float, device: torch.device = 'CPU'):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to CPU.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

class CNN(nn.Module):
    def __init__(self, input_layer, hidden_layers, output_layer):
        super().__init__()
        self.block1 = nn.Sequential(
        nn.Conv2d(input_layer, hidden_layers, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = nn.Sequential(
        nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(hidden_layers * 7 * 7, output_layer)
        )

    def forward(self, x):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        out = self.classifier(block2_out)
        return out

# Setup training data
train_data = datasets.FashionMNIST(
    root="./Images", # where to download data to?
    train=True, # get training data
    download=False, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="./Images",
    train=False, # get test data
    download=False,
    transform=ToTensor()
)
# Print informations about the datatset
image, label = train_data[0] # Each element is a tuple of an image and its label
# print("Shape of a single image:", image.shape)
# How many samples are there? 
# print(f"Number of Training Images: {len(train_data.data)}, Test Images: {len(test_data.data)}")
# train_data.targets gives the number of targets, which is equal to the data since each image has a single target
class_names  = train_data.classes # gives the classes(targets) names (a list of 10 elements)

# Visualize the image
# plt.imshow(image.squeeze(), cmap= "gray") # image shape is [1, 28, 28] (colour channels, height, width)
# plt.title(class_names[label])
# plt.show()

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Let's check out what we've created
# print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
# print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")


# Initialize the model, loss, and optimizer

hidden_layers = 10

Model = CNN(1, hidden_layers, 10) # Input is the number of color channels, Output are the classes
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params= Model.parameters(), lr= 0.1)

# Training the Model
epochs = 3
train_time_start_on_cpu = timer()
for i in tqdm(range(epochs)):
    print(f"Epoch: {i+1}\n-------")
    train_loss_per_batch = 0
    for batch, (X, y) in enumerate(train_dataloader):
        Model.train()
        output = Model(X)
        loss = loss_fn(output, y)
        train_loss_per_batch += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss_per_batch /= len(train_dataloader)

    ### Testing the Model along training
    # Setup variables for accumulatively adding up loss
    test_loss = 0
    Model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = Model(X)
           
            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss_per_batch:.5f} | Test loss: {test_loss:.5f}\n")

# Calculate training time      
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu)
