import torch.nn as nn
import torch.nn.functional as F


class LinModel(nn.Module):
    def __init__(self, 
                 geometry,
                 spectrum,
                 hidden_layers = [1000, 1000, 1000, 1000, 1000, 1000, 1000]
                 ):
        super(LinModel, self).__init__()
        self.bp = False
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        self.linears.append(nn.Linear(geometry, hidden_layers[0]))
        self.bn_linears.append(nn.BatchNorm1d(hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            self.linears.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.bn_linears.append(nn.BatchNorm1d(hidden_layers[i+1]))
        self.linears.append(nn.Linear(hidden_layers[-1], spectrum))

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
        S = self.linears[-1](out)
        
        return S