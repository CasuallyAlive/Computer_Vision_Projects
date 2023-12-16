import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    ###########################################################################
    # Student code begin
    ###########################################################################
    alexnet_model = alexnet(pretrained=True)
    # print(alexnet_model)
    
    # cnn layers frm alexnet
    an_cnn_layers = alexnet_model.features    
    for param in an_cnn_layers.parameters():
      param.requires_grad = False
    self.cnn_layers = an_cnn_layers

    # avgpool layer from alexnet
    an_avgpool_layers = alexnet_model.avgpool
    for param in an_avgpool_layers.parameters():
      param.requires_grad = False
    self.avgpool = an_avgpool_layers
        
    # Fully connected layers from alexnet
    an_fc_layers = alexnet_model.classifier
    for param in an_fc_layers.parameters():
      param.requires_grad = False
    num_in_features = an_fc_layers[-3].out_features 
    # print(num_in_features)
    an_fc_layers[-1] = nn.Dropout(p=0.5)
    an_fc_layers.add_module(name="7", 
                module=nn.Linear(in_features=num_in_features, out_features=15))
    
    self.fc_layers = an_fc_layers
    
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
    x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images

    ###########################################################################
    # Student code begin
    ###########################################################################
    
    cnn_out = self.avgpool(self.cnn_layers(x))
    cnn_out = cnn_out.view(-1, 256 * 6 * 6)
    model_output = self.fc_layers(cnn_out)

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output