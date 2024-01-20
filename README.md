# Develop-Image-Classifier-for-Flowers-with-Deep-Learning

An image classifier is trained to recognize different species of flowers with 102 flowers categories provided.
A pretrained model from torchvision is used, a new untrained feed-forward network is defined as a classifier
using ReLU activations.


The project is broken down into multiple steps:
  i. Load and preprocess the image dataset.
  ii. Train the image classifier on dataset.
  iii. Track the loss and accuracy on the validation set to determine the best hyperparameters.
  iv. Use the trained classifier to predict image content.

Furthermore, python scripts such as train.py and predict.py are created and can be run as an application
to train a neural network and predict class of flower by accepting input of a file path. The application is run
by invoking the file using the command "python train.py" and "python predict.py".


