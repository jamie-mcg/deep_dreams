from PIL import Image as img
from torchvision import transforms
import torch
import numpy as np



def features(image, model, layers):
        features = {}
        for layer_n, layer in model.get_layers():
            image = layer(image)
            if layer_n in layers.keys():
                features[layers[layer_n]] = image

        return features