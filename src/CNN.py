# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms, utils
from torchvision.transforms import functional
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import matplotlib.pyplot as plt
# %%
above_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_path = above_path + '/input_fig/10001/plot_10001-0.png'
# %%
train_data_dir = '/Users/abnerteng/Desktop/stock_fig/10051/10051_train'
test_data_dir = '/Users/abnerteng/Desktop/stock_fig/10051/10051_test'

train_dataset = ImageFolder(train_data_dir, transform = transforms.ToTensor())

test_dataset = ImageFolder(test_data_dir, transform = transforms.ToTensor())
# %%
img, label = train_dataset[0]
print(img.shape)
# %%
def initialize(self, data_path, split = "train", transform = NULL, traget_transform = NULL, indexes = NULL):
    self.transform = transform
    self.target_transform = target_transform

    if split == "train":
        self.images = data_path + 'train.csv'
        if (indexes != NULL):
            self.images = self.images[indexes, ]
        self.path = data_path + 'train_images'
    elif split == "test":
        self.images = data_path + 'test.csv'
        if (indexes != NULL):
            self.images = self.images[indexes, ]
        self.path = data_path + 'test_images'
# %%
transform = transforms.ToTensor()
img = Image.open(img_path)
img_tensor = transform(img)[:3, :, :]
PIL_tensor = functional.to_pil_image(img_tensor)
gs_img = torchvision.transforms.functional.to_grayscale(PIL_tensor, num_output_channels=1)

# %%
## CNN model: with 20 days scale, 5x3  convolutional layer with 60 channels, 2x1 max-pooling layer, 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        ## image shape is 1*60*60 where 1 is the color channels
        ## convolutional layer
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (5, 3), stride = (3,1))
        self.LR = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size = (2, 1), stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (5, 3), stride = (3,1))
        self.LR = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size = (2, 1), stride = 1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (5, 3), stride = (3,1))
        self.LR = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size = (2, 1), stride = 1)
        ## flatten layer
        self.flatten = nn.Flatten()
        ## linear layer
        self.fc1 = nn.Linear(46080, )
        self.sm = nn.softmax()

    def forward(self, x):
        x = self.conv1
        x = self.LR(x)
        x = self.pool(x)
        x = self.conv2
        x = self.LR(x)
        x = self.pool(x)
        x = self.conv3
        x = self.LR(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        output = self.sm(x)
        return output
# %%
