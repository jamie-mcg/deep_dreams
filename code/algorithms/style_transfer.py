import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("..")

from image_processing import convert_image

OPTIMIZERS = {
    "adam": torch.optim.Adam
}

class StyleTransfer(nn.Module):
    def __init__(self, model, content_image, style_image, layers, style_weightings, content_layer="conv4_2", 
                    content_weight=1, style_weight=1e6, optimizer="adam", lr=0.003, save_path=None):
        super(StyleTransfer, self).__init__()

        self._model = model
        self._layers = layers

        self._content_image = content_image
        self._content_features = self.features(self._content_image)
        self._content_weight = content_weight
        self._content_layer = content_layer

        self._style_image = style_image
        self._style_features = self.features(self._style_image)
        self._style_grams = {layer_name: self.gram_matrix(layer) for layer_name, layer in self._style_features.items()}
        self._style_weight = style_weight
        self._style_weightings = style_weightings

        self._art = content_image.clone().requires_grad_(True)

        self._lr = lr

        self._optimizer = OPTIMIZERS[optimizer.lower()]([self._art], lr=self._lr)

        self._show_every = 5
        self._save_path = save_path
        os.mkdir(save_path)

    def content_loss(self, features):
        return torch.mean((features[self._content_layer] - self._content_features[self._content_layer])**2)

    def gram_matrix(self, filter_resp):
        filter_resp = filter_resp.view(filter_resp.size()[1], -1)

        return torch.mm(filter_resp, filter_resp.t())

    def style_loss(self, gram_a, gram_b):
        return torch.mean((gram_a - gram_b)**2)

    def style_loss_total(self, features):
        loss = 0
        for layer_name, layer_weight in self._style_weightings.items():
            feature = features[layer_name]
            shape = feature.shape

            gram_matrix_a = self.gram_matrix(feature)
            gram_matrix_b = self._style_grams[layer_name]

            norm = shape[1] * shape[2] * shape[3]
            loss += self._style_weightings[layer_name] * self.style_loss(gram_matrix_a, gram_matrix_b) / norm

        return loss

    def aggregate_loss(self, features):
        return self._content_weight * self.content_loss(features) + self._style_weight * self.style_loss_total(features)


    def features(self, image):
        features = {}
        for layer_n, layer in self._model.get_layers():
            image = layer(image)
            if layer_n in self._layers.keys():
                features[self._layers[layer_n]] = image

        return features

    def forward(self, iterations=500):
        for iteration in range(iterations):
            self._optimizer.zero_grad()

            art_features = self.features(self._art)

            total_loss = self.aggregate_loss(art_features)

            total_loss.backward()
            self._optimizer.step()

            if iteration % self._show_every == 0:
                print(f"Total Loss: {float(total_loss)}")
                plt.imshow(convert_image(self._art))
                if self._save_path:
                    plt.savefig(os.path.join(self._save_path, f"img_{iteration}"))
                plt.show()

