from torchvision import transforms
import numpy as np

normalisation = [(0.485, 0.456, 0.406), 
                 (0.229, 0.224, 0.225)]

def transform(image, max_size=400):
    image = image.convert("RGB")

    if max(image.size) < max_size:
        max_size = max(image.size)

    transform = transforms.Compose([transforms.Resize(max_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(normalisation[0],
                                                         normalisation[1])])

    return transform(image)[:3,:,:].unsqueeze(0)

def convert_image(img):
    image = img.to("cpu").clone().detach().numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image*np.array(normalisation[1]) + np.array(normalisation[0])
    image = image.clip(0, 1)
    return image
