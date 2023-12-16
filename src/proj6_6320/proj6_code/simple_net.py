import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################
    # def calc_out_shape_conv(N, Ci, Hi, Wi, Co, k, s, p=0, d=1):
    #   Ho = ((Hi + 2*p-d*(k-1)-1)/s)+1
    #   Wo = ((Wi + 2*p-d*(k-1)-1)/s)+1
    #   return N, Co, Ho, Wo
    
    # DEF_CONV_K_SZ = 5; DEF_CONV_STRIDE = 1; DEF_FC_OUT = 120
    
    # experiment = [True, False, False]
    # exp_k = 4; exp_s = 3; exp_fc = 180
    
    # conv_k = DEF_CONV_K_SZ if not experiment[0] else exp_k
    # conv_s = DEF_CONV_STRIDE if not experiment[1] else exp_s
    # fc = DEF_FC_OUT if not experiment[2] else exp_fc
    
    self.cnn_layers = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0), # Conv: 5 x 5 kernel. 1@64x64 -> 10@60x60
      nn.MaxPool2d(kernel_size=3, stride=3), # Pool 3 x 3 kernel. 10@60x60 -> 10@20x20
      nn.ReLU(),
      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0), # Conv: 5 x 5 kernel. 10@20x20 -> 20@16x16
      nn.MaxPool2d(kernel_size=3, stride=3), # Pool 3 x 3 kernel. 20@16x16 -> 20@5x5
      nn.ReLU()                    
    )
    self.fc_layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=500, out_features=120),
      nn.ReLU(),
      nn.Linear(in_features=120, out_features=15)
    )
    
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    model_output = self.cnn_layers(x)
    model_output = self.fc_layers(model_output)
    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output

  # Alternative implementation using functional form of everything
class SimpleNet2(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 10, 5)
      self.pool = nn.MaxPool2d(3, 3)
      self.conv2 = nn.Conv2d(10, 20, 5)
      self.fc1 = nn.Linear(500, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 15)
      self.loss_criterion = nn.CrossEntropyLoss()

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 500)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x