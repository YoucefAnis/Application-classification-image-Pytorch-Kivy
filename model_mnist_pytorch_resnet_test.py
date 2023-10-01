import pickle
import gzip
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 20, 5, 1) # Conv layer1
      self.conv2 = nn.Conv2d(20, 50, 5, 1) # Conv layer2
      self.fc1 = nn.Linear(4*4*50, 500)    # Fully connected layer1
      self.dropout1 = nn.Dropout(0.5)   # We use dropout layer between these both FCL as they have the highest number of parameters b/t them
      self.fc2 = nn.Linear(500, 10)   # Fully connected layer2
    def forward(self, x):
      x = F.relu(self.conv1(x))  # Apply ReLu to the feature maps produced after Conv 1 layer
      x = F.max_pool2d(x, 2, 2)  # Pooling layer after Conv 1 layer
      x = F.relu(self.conv2(x))  # Apply ReLu to the feature maps produced after Conv 2 layer
      x = F.max_pool2d(x, 2, 2)  # Pooling layer after Conv 2 layer
      x = x.view(-1, 4*4*50)     # Flattening the output of CNN to feed it into Fully connected layer
      x = F.relu(self.fc1(x))   # Fully connected layer 1 with Relu
      x = self.dropout1(x)     # We use dropout layer between these both FCL as they have the highest number of parameters b/t them
      x = self.fc2(x)         # Fully connected layer 2 with no activation funct as we need raw output from CrossEntropyLoss
      return x
    

transform2 = transforms.Compose([transforms.Resize((28,28)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                               ])

model3 = LeNet().to(device)
model3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model3.parameters(), lr = 0.0001)
def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy() # Just use cpu in this case as it is not compatible with gpu
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image