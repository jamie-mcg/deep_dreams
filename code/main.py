import matplotlib.pyplot as plt
from PIL import Image as img

from image_processing import transform, convert_image

from models import VGG
from algorithms import StyleTransfer


if __name__ == "__main__":
    content_path = "../images/content/neil.jpg"
    style_path = "../images/style/kadinsky.jpg"

    content_image = transform(img.open(content_path))
    style_image = transform(img.open(style_path))

    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(convert_image(content_image))
    # plt.show()

    vgg = VGG()

    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1', 
        '10': 'conv3_1', 
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }

    style_weights = {
        "conv1_1": 1.0,
        "conv2_1": 0.75,
        "conv3_1": 0.2,
        "conv4_1": 0.2,
        "conv5_1": 0.2
    }

    model = StyleTransfer(vgg, content_image, style_image, layers, style_weights, save_path="../images/kadineil")

    model.forward()


