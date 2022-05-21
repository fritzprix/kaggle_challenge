from typing import Tuple
import torch
from torch import nn
from torchvision import transforms as T


KLS = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]


class InceptionBlock(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernels=[3, 5]):
        super().__init__()
        self.kerenls = kernels
        for idx, k in enumerate(kernels):
            if k % 2 == 0:
                raise 'kernel size should be odd number'
            self._modules[f'Conv Ch.{idx}_K_{k}X{k}'] = nn.Conv2d(input_channels, output_channels, kernel_size=k, padding=(k // 2))
        self._modules[f'MxPool_Kx3'] = nn.Sequential(
            nn.MaxPool2d(3, padding=1, stride=1), 
            nn.Conv2d(input_channels, output_channels, kernel_size=1))
        self._modules[f'Conv1X1'] = nn.Conv2d(input_channels, output_channels, kernel_size=1)
    
    def forward(self, X):
        Y = []
        for idx, k in enumerate(self.kerenls):
            Y.append(self._modules[f'Conv Ch.{idx}_K_{k}X{k}'](X))
        Y.append(self._modules[f'MxPool_Kx3'](X))
        Y.append(self._modules[f'Conv1X1'](X))
        Y.append(X)
        return torch.cat(Y,1)


from torchvision.transforms import functional as F
from PIL.Image import Image
class Padding(object):
    
    def __init__(self, size: Tuple):
        super(Padding).__init__()
        self.size = size

    def __call__(self, X: Image):
        ## torch compatible image fromat assumed (N,C,H,W)
        h, w = self.size
        iw, ih = X.size
        if ih == h and iw == w:
            return X
        else:
            tp = (h - ih) // 2
            bp = h - tp - ih
            lp = (w - iw) // 2
            rp = w - lp - iw
            return F.pad(X, [lp, tp, rp, bp])

class DenseInception_V0(nn.Sequential):
    
    def __init__(self):
        super().__init__(InceptionBlock(3, 8),
                        nn.BatchNorm2d(8 * 4 + 3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),         ## (n,256,75,75)
                        nn.Conv2d(8 * 4 + 3, 8 * 2, kernel_size=1),      ## (n,128, 75, 75)
                        InceptionBlock(8 * 2, 8 * 2),                   ## (n,196 * 4, 75, 75)  
                        nn.BatchNorm2d(8 * 2 * 5),                       
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=3),        ## (n, 196 * 4, 25, 25)
                        nn.Conv2d(8 * 2 * 5, 8 * 5, kernel_size=1),   ## (n, 196 * 2, 25, 25)
                        InceptionBlock(8 * 5, 8 * 5),             ## (n, 196 * 12, 25, 25)
                        nn.BatchNorm2d(8 * 5 * 5),                  ## (n, )
                        nn.ReLU(),
                        nn.AvgPool2d(kernel_size=3, padding=1),
                        nn.Flatten(),
                        nn.Linear(8 * 5 * 5 * 9 * 9, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 512),
                        nn.ReLU(),
                        nn.Linear(512, len(KLS)))

