from torchvision import transforms
import numpy as np

normalisation = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

def transform(image):
    image = image.convert("RGB")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(normalisation[0],
                                                         normalisation[1])])

    return transform(image).unsqueeze(0)

def convert_image(img):
    image = img.to("cpu").clone().detach().numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image*np.array(normalisation[0]) + np.array(normalisation[1])
    image = image.clip(0, 1)
    return image
