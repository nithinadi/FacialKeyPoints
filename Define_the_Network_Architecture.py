#!/usr/bin/env python
# coding: utf-8

# ## Define the Convolutional Neural Network
# 
# After you've looked at the data you're working with and, in this case, know the shapes of the images and of the keypoints, you are ready to define a convolutional neural network that can *learn* from this data.
# 
# In this notebook and in `models.py`, you will:
# 1. Define a CNN with images as input and keypoints as output
# 2. Construct the transformed FaceKeypointsDataset, just as before
# 3. Train the CNN on the training data, tracking loss
# 4. See how the trained model performs on test data
# 5. If necessary, modify the CNN structure and model hyperparameters, so that it performs *well* **\***
# 
# **\*** What does *well* mean?
# 
# "Well" means that the model's loss decreases during training **and**, when applied to test image data, the model produces keypoints that closely match the true keypoints of each face. And you'll see examples of this later in the notebook.
# 
# ---
# 

# ## CNN Architecture
# 
# Recall that CNN's are defined by a few types of layers:
# * Convolutional layers
# * Maxpooling layers
# * Fully-connected layers
# 
# You are required to use the above layers and encouraged to add multiple convolutional layers and things like dropout layers that may prevent overfitting. You are also encouraged to look at literature on keypoint detection, such as [this paper](https://arxiv.org/pdf/1710.00977.pdf), to help you determine the structure of your network.
# 
# 
# ### TODO: Define your model in the provided file `models.py` file
# 
# This file is mostly empty but contains the expected name and some TODO's for creating your model.
# 
# ---

# ## PyTorch Neural Nets
# 
# To define a neural network in PyTorch, you define the layers of a model in the function `__init__` and define the feedforward behavior of a network that employs those initialized layers in the function `forward`, which takes in an input image tensor, `x`. The structure of this Net class is shown below and left for you to fill in.
# 
# Note: During training, PyTorch will be able to perform backpropagation by keeping track of the network's feedforward behavior and using autograd to calculate the update to the weights in the network.
# 
# #### Define the Layers in ` __init__`
# As a reminder, a conv/pool layer may be defined like this (in `__init__`):
# ```
# # 1 input image channel (for grayscale images), 32 output channels/feature maps, 3x3 square convolution kernel
# self.conv1 = nn.Conv2d(1, 32, 3)
# 
# # maxpool that uses a square window of kernel_size=2, stride=2
# self.pool = nn.MaxPool2d(2, 2)      
# ```
# 
# #### Refer to Layers in `forward`
# Then referred to in the `forward` function like this, in which the conv1 layer has a ReLu activation applied to it before maxpooling is applied:
# ```
# x = self.pool(F.relu(self.conv1(x)))
# ```
# 
# Best practice is to place any layers whose weights will change during the training process in `__init__` and refer to them in the `forward` function; any layers or functions that always behave in the same way, such as a pre-defined activation function, should appear *only* in the `forward` function.

# #### Why models.py
# 
# You are tasked with defining the network in the `models.py` file so that any models you define can be saved and loaded by name in different notebooks in this project directory. For example, by defining a CNN class called `Net` in `models.py`, you can then create that same architecture in this and other notebooks by simply importing the class and instantiating a model:
# ```
#     from models import Net
#     net = Net()
# ```

# In[27]:


# load the data if you need to; if you have already loaded the data, you may comment this cell out
# -- DO NOT CHANGE THIS CELL -- #

# get_ipython().system('mkdir /data')
# get_ipython().system('wget -P /data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip')
# get_ipython().system('unzip -n /data/train-test-data.zip -d /data')


# <div class="alert alert-info">**Note:** Workspaces automatically close connections after 30 minutes of inactivity (including inactivity while training!). Use the code snippet below to keep your workspace alive during training. (The active_session context manager is imported below.)
# </div>
# ```
# from workspace_utils import active_session
# 
# with active_session():
#     train_model(num_epochs)
# ```
# 

# In[1]:


# import the usual resources
import matplotlib.pyplot as plt
import numpy as np

# import utilities to keep workspaces alive during model training
# from workspace_utils import active_session

# watch for any changes in model.py, if it changes, re-load it automatically
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


## TODO: Define the Net in models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from models import Net

net = Net()
# net.cuda()
print(net)


# ## Transform the dataset 
# 
# To prepare for training, create a transformed dataset of images and keypoints.
# 
# ### TODO: Define a data transform
# 
# In PyTorch, a convolutional neural network expects a torch image of a consistent size as input. For efficient training, and so your model's loss does not blow up during training, it is also suggested that you normalize the input images and keypoints. The necessary transforms have been defined in `data_load.py` and you **do not** need to modify these; take a look at this file (you'll see the same transforms that were defined and applied in Notebook 1).
# 
# To define the data transform below, use a [composition](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms) of:
# 1. Rescaling and/or cropping the data, such that you are left with a square image (the suggested size is 224x224px)
# 2. Normalizing the images and keypoints; turning each RGB image into a grayscale image with a color range of [0, 1] and transforming the given keypoints into a range of [-1, 1]
# 3. Turning these images and keypoints into Tensors
# 
# These transformations have been defined in `data_load.py`, but it's up to you to call them and create a `data_transform` below. **This transform will be applied to the training data and, later, the test data**. It will change how you go about displaying these images and keypoints, but these steps are essential for efficient training.
# 
# As a note, should you want to perform data augmentation (which is optional in this project), and randomly rotate or shift these images, a square image size will be useful; rotating a 224x224 image by 90 degrees will result in the same shape of output.

# In[3]:



from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor


## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(227),
                                    RandomCrop(224),
                                    Normalize(),
                                    ToTensor()])

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'


# In[5]:


# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='D:/Computer_vision_udacity/P1_Facial_Keypoints-master/data/training_frames_keypoints.csv',
                                             root_dir='D:/Computer_vision_udacity/P1_Facial_Keypoints-master/data/training/',
                                             transform=data_transform)


print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size(),sample['keypoints'])


# ## Batching and loading data
# 
# Next, having defined the transformed dataset, we can use PyTorch's DataLoader class to load the training data in batches of whatever size as well as to shuffle the data for training the model. You can read more about the parameters of the DataLoader, in [this documentation](http://pytorch.org/docs/master/data.html).
# 
# #### Batch size
# Decide on a good batch size for training your model. Try both small and large batch sizes and note how the loss decreases as the model trains. Too large a batch size may cause your model to crash and/or run out of memory while training.
# 
# **Note for Windows users**: Please change the `num_workers` to 0 or you may face some issues with your DataLoader failing.

# In[6]:


# load training data in batches
batch_size = 10

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)


# ## Before training
# 
# Take a look at how this model performs before it trains. You should see that the keypoints it predicts start off in one spot and don't match the keypoints on a face at all! It's interesting to visualize this behavior so that you can compare it to the model after training and see how the model has improved.
# 
# #### Load in the test dataset
# 
# The test dataset is one that this model has *not* seen before, meaning it has not trained with these images. We'll load in this test data and before and after training, see how your model performs on this set!
# 
# To visualize this test data, we have to go through some un-transformation steps to turn our images into python images from tensors and to turn our keypoints back into a recognizable range. 

# In[7]:


# load in the test data, using the dataset class
# AND apply the data_transform you defined above

# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='D:/Computer_vision_udacity/P1_Facial_Keypoints-master/data/test_frames_keypoints.csv',
                                             root_dir='D:/Computer_vision_udacity/P1_Facial_Keypoints-master/data/test/',
                                             transform=data_transform)
print('Number of test images: ', len(test_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = test_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size(),sample['keypoints'])


# In[8]:


# load test data in batches
batch_size = 10

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)

# print("test_loader",test_loader)
# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=4)

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
# Helper function to show a batch

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break





