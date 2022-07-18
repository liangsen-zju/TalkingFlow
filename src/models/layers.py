import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Follow: 
https://github.com/PolinaKirichenko/flows_ood/blob/master/flow_ssl/glow/glow_utils.py
https://github.com/tatsy/normalizing-flows-pytorch/blob/master/flows/modules.py
"""

class Invertible1x1Conv1d(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, num_channels):
        super(Invertible1x1Conv1d, self).__init__()

        self.num_channels = num_channels
        weight = torch.qr(torch.randn((num_channels, num_channels)))[0]
        self.weight = nn.Parameter(weight)              # (num_channels, num_channels)

    def forward(self, input, logdet=None):
        """
        log-det = log|abs(|W|)| * nseq
        """

        nbatch, nchannels, nseq = input.size()    # (B,C,N)
        slogdet = (torch.slogdet(self.weight)[1] * nseq).expand(nbatch)    # w * h * log |det w|
        
        weight = self.weight.view(self.num_channels, self.num_channels, 1)
        z = F.conv1d(input, weight)
        
        logdet = logdet + slogdet if (logdet is not None) else slogdet

        return z, logdet
    
    def inverse(self, input, logdet=None):
        """
        log-det = log|det(W^-1)| * nseq = - log|det(W)| * nseq 
        """
        nbatch, nchannels, nseq = input.size()    # (B,C,N)
        slogdet = - (torch.slogdet(self.weight)[1] * nseq).expand(nbatch)    # w * h * log |det w|
        
        weight = torch.inverse(self.weight)
        weight = weight.view(self.num_channels, self.num_channels, 1)
        z = F.conv1d(input, weight)

        logdet = logdet + slogdet if (logdet is not None) else slogdet

        return z, logdet


class Invertible1x1Conv2d(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, num_channels):
        super(Invertible1x1Conv2d).__init__()

        self.num_channels = num_channels
        weight = torch.qr(torch.randn((num_channels, num_channels)))[0]
        self.weight = nn.Parameter(weight)   # (num_channels, num_channels)

    def forward(self, input, logdet=None):
        """
        log-det = log|abs(|W|)| * h * w
        """
        nbatch, nchannels, h, w = input.size()    # (B, C, H, W)
        slogdet = (torch.slogdet(self.weight)[1] * h * w).expand(nbatch)    # w * h * log |det w|
        
        weight = self.weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(input, weight)
        logdet = logdet + slogdet if (logdet is not None) else None

        return z, logdet
    
    def inverse(self, input, logdet=None):
        """
        log-det = log|det(W^-1)| * h * w = - log|det(W)| * h * w 
        """
        nbatch, nchannels, h, w = input.size()    # (B,C,N)
        slogdet = - (torch.slogdet(self.weight)[1] * h * w).expand(nbatch)    # w * h * log |det w|
        
        weight = torch.inverse(self.weight)
        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(input, weight)
        logdet = logdet + slogdet if (logdet is not None) else None

        return z, logdet


class ActNorm1d(nn.Module):
    """ Initializ at the first minibatch with zero mean and unit variance, 
    After initialization, the bias and logs are treated as a regular trainable parameters that are dependent of data.
    """
    def __init__(self, num_channels, eps=1e-6):
        super(ActNorm1d, self).__init__()

        self.inited = False
        self.eps = eps
        self.num_channels = num_channels

        # register bias and scale 
        self.bias = nn.Parameter(torch.zeros((1, num_channels, 1)))
        self.logs = nn.Parameter(torch.zeros((1, num_channels, 1)))   # log scale, which easy for det(log|s|)

    def init_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = torch.mean(input, dim=[0,2], keepdim=True)                       # (1, C, 1)
            logs = torch.log(torch.std(input, dim=[0,2], keepdim=True) + self.eps)  # (1, C, 1)
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True
            
            
    def forward(self, input, logdet=None):
        B,C,N = input.size()

        if not self.inited:
            self.init_parameters(input)
        
        z = (input - self.bias) / torch.exp(self.logs)    # Normalize by element-wise
        dlogdet = - torch.sum(self.logs) * N
        logdet = logdet + dlogdet if (logdet is not None) else None
        
        return z, logdet

    def inverse(self, input, logdet=None):
        B,C,N = input.size()

        z = input * torch.exp(self.logs) + self.bias
        dlogdet = torch.sum(self.logs) * N
        logdet = logdet + dlogdet if (logdet is not None) else None
        return z, logdet


class ActNorm2d(nn.Module):
    """ Initializ at the first minibatch with zero mean and unit variance, 
    After initialization, the bias and logs are treated as a regular trainable parameters that are dependent of data.
    """
    def __init__(self, num_channels, eps=1e-6):
        super(ActNorm2d, self).__init__()

        self.inited = False
        self.eps = eps
        self.num_channels = num_channels

        # register bias and scale 
        self.bias = nn.Parameter(torch.zeros((1, num_channels, 1, 1)))
        self.logs = nn.Parameter(torch.zeros((1, num_channels, 1, 1)))   # log scale, which easy for det(log|s|)

    def init_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = torch.mean(input, dim=[0,2,3], keepdim=True)            # (1, C, 1, 1)
            logs = torch.log(torch.std(input, dim=[0,2,3], keepdim=True) + self.eps)  # (1, C, 1, 1)
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def forward(self, input, logdet=None):
        B,C,H,W = input.size()

        if not self.inited:
            self.init_parameters(input)
        
        z = (input - self.bias) / torch.exp(self.logs)
        dlogdet = - torch.sum(self.logs) * H * W
        logdet = logdet + dlogdet if (logdet is not None) else None

        return z, logdet

    def inverse(self, input, logdet=None):
        B,C,H,W = input.size()

        z = input * torch.exp(self.logs) + self.bias
        dlogdet = torch.sum(self.logs) * H * W
        logdet = logdet + dlogdet if (logdet is not None) else None

        return z, logdet