import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, 
                 geometry,
                 spectrum,
                 num_linear_layers = 4,
                 num_conv_layers = 3,
                 num_linear_neurons = 1000,
                 num_conv_out_channel = 4,
                 ):
        super(BaseModel, self).__init__()
        self.bp = False
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        self.linears.append(nn.Linear(geometry, num_linear_neurons))
        self.bn_linears.append(nn.BatchNorm1d(num_linear_neurons))
        for _ in range(num_linear_layers - 1):
            self.linears.append(nn.Linear(num_linear_neurons, num_linear_neurons))
            self.bn_linears.append(nn.BatchNorm1d(num_linear_neurons))
        self.linears.append(nn.Linear(num_linear_neurons, (spectrum // 2)))
        self.bn_linears.append(nn.BatchNorm1d(spectrum // 2))

        # Conv Layer definitions here
        self.convs = nn.ModuleList([])
        in_channel = 1
        self.convs.append(nn.ConvTranspose1d(1, num_conv_out_channel, 2, stride=2, padding=0))
        
        # Additional conv layers (if requested)
        in_channel = num_conv_out_channel
        for _ in range(1, num_conv_layers):
            kernel_size = 5
            pad = int((kernel_size - 1)/2)
            
            self.convs.append(nn.ConvTranspose1d(in_channel, num_conv_out_channel, 
                              kernel_size, stride=1, padding=pad))
            in_channel = num_conv_out_channel
        
        if num_conv_layers > 0:
            self.convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, 
                             stride=1, padding=0))
    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The spectrum-dimension output
        """
        out = G 
        
        # For the linear part
        for _, (fc, bn) in enumerate(zip(self.linears[:-1], self.bn_linears)):
            out = F.relu(bn(fc(out))) 
        
        # Final linear layer (no batch norm)
        out = self.linears[-1](out)
        
        # Reshape for convolution
        out = out.unsqueeze(1)
        
        # For the conv part
        for ind, conv in enumerate(self.convs[:-1]):
            out = F.relu(conv(out))
        
        # Final conv layer (no activation)
        if len(self.convs) > 0:
            out = self.convs[-1](out)
            
        # Remove channel dimension if needed
        S = out.squeeze(1)
        
        return S