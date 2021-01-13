import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(VGG, self).__init__()
        self._vgg = models.vgg19(pretrained=pretrained).features

        if freeze:
            for param in self._vgg.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        return self._vgg.forward(x)

    def get_layers(self):
        return self._vgg._modules.items()