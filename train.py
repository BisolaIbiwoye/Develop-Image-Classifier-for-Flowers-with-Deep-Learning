# Import packages needed

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import argparse
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Create Parse using ArgumentParser
parser = argparse.ArgumentParser()

# Create command line arguments as mentioned above using add_argument() from ArguementParser method
parser.add_argument('--data_dir', type = str, default = './flowers/', help = 'path to the folder of flower images')
parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Model Architecture')
parser.add_argument('--save_dir', type = str, default = '/checkpoint.pth', help = 'where to save checkpoint')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate to train model')
parser.add_argument('--epochs', type = int, default = 3, help = 'the number of complete pass of the training data set                              through the neural network')
parser.add_argument('--gpu', type = str, default = 'gpu', help = 'allow user to use a faster processor for training')

args = parser.parse_args()
arch = args.arch
# Data directory

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([ transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


# Using the image datasets and the transforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


#download module
def load_model(arch='vgg16'):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        arch == 'alexnet'
        model = models.alexnet(pretrained=True)
    
    #freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #function for new classifier
    my_classifier = nn.Sequential (OrderedDict([
                        ('fc1', nn.Linear(25088, 512)),
                        ('relu1', nn.ReLU()),
                        ('fc2', nn.Linear(512, 256)),
                        ('relu2', nn.ReLU()),
                        ('fc3', nn.Linear(256, 128)),
                        ('relu3', nn.ReLU()),
                        ('fc4', nn.Linear(128, 102)),
                        ('output', nn.LogSoftmax (dim=1))
                        ]))
    model.classifier = my_classifier
    return model, arch


model, arch = load_model('vgg16')

# Use GPU if it's available
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(gpu);

# funtion to train model
epochs = 3
def train_model(model = model, trainloader = trainloader, criterion = criterion, validloader = validloader, optimizer =         optimizer, epochs = epochs, gpu = gpu):
    steps = 0
    print_every = 40
    train_losses = []
    validation_losses = []
    for epoch in range(epochs):
        running_loss = 0

        for images, labels in trainloader:
            steps += 1
            #move input and label tensors to the default device
            images, labels = images.to(gpu), labels.to(gpu)

            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            if steps % print_every == 0:
                
                #move model to evaluation mode
                model.eval()

                validation_loss = 0
                accuracy = 0

                # Turn off gradient for validation.
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(gpu), labels.to(gpu)
                        outputs = model.forward(images)
                        validation_loss = criterion(outputs, labels)




                        #calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))


                train_losses.append(running_loss/len(trainloader))
                validation_losses.append(validation_loss/len(validloader))


                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}.. ")
                running_loss = 0

                #change to train mode
                model.train()

#Train model
train_model(model, trainloader, criterion, validloader, optimizer, epochs, gpu)

# Do validation on the test

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_check(testloader = testloader, gpu = gpu):
    correct = 0
    total = 0

    # Turn off gradient for validation.
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(gpu), labels.to(gpu)
            output = model(images)

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy is: {(correct/total) * 100}%')

# run a check
test_check(testloader, gpu)

 #get class to idx
class_to_idx = train_data.class_to_idx

#define checkpoints

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': arch,
              'state_dict' : model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict(),
              'optimizer': optimizer,
              'criterion': criterion,
              'classifier' : model.classifier,
              'epochs' : epochs,
              'class_to_idx' : class_to_idx}

# Save the checkpoint 
torch.save(checkpoint, 'checkpoint.pth')