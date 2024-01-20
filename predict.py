# Import packages needed

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import argparse
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    
# Create Parse using ArgumentParser
parser = argparse.ArgumentParser()

# Create command line arguments as mentioned above using add_argument() from ArguementParser method
parser.add_argument('--test_file', type = str, default = 'flowers/test/10/image_07090.jpg', help = 'path to a flower image')
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'where to access checkpoint to load model                        for prediction')
parser.add_argument('--json_file', type = str, default = 'cat_to_name.json' , help = 'a dictionary mapping the integer                               encoded categories to the actual names of the flowers.')
parser.add_argument('--TopK', type = int, default = 5, help = 'the Top "K" probabilities for prediction')
parser.add_argument('--gpu', type = str, default = 'gpu', help = 'allow user to use a faster processor for training')

args = parser.parse_args()
test_file = args.test_file

# function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    arch = checkpoint['arch']
    classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    criterion = checkpoint['criterion']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)

    model.class_to_idx =  class_to_idx
    model.classifier = classifier
    model.load_state_dict(state_dict)
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model
    
model = load_checkpoint('checkpoint.pth')
#Function to process image

def process_image(image=test_file):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    transform_image = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    np_image = transform_image(img).float()
    return np_image

    
    
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Function for class prediction
def predict(image_path=test_file, model=model, topk=5, gpu=gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = image.float().unsqueeze_(0)
    # Turn off gradient for validation.
    
    model.to(gpu)
    with torch.no_grad():
        image = image.to(gpu)
        output = model.forward(image)
    prediction = F.softmax(output.data, dim = 1)


    probs, indices = prediction.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]

    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]

    return probs, classes

probs, classes = predict(test_file, model)
flower_names = [cat_to_name [item] for item in classes]
print(probs)
print(classes)
print(flower_names)

