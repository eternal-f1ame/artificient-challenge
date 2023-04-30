"""
Use the following augmentation methods on the sample image under data/sample.png
and save the result under this path: 'data/sample_augmented.png'

Note:
    - use torchvision.transforms
    - use the following augmentation methods with the same order as below:
        * affine: degrees: ±5, 
                  translation= 0.1 of width and height, 
                  scale: 0.9-1.1 of the original size
        * rotation ±5 degrees,
        * horizontal flip with a probablity of 0.5
        * center crop with height=320 and width=640
        * resize to height=160 and width=320
        * color jitter with:  brightness=0.5, 
                              contrast=0.5, 
                              saturation=0.4, 
                              hue=0.2
    - use default values for anything unspecified
"""

import torch
from torchvision import transforms as T
import numpy as np
import cv2


torch.manual_seed(8)
np.random.seed(8)

img = cv2.imread('data/sample.png')

# write your code here ...

def task2(img):
    """Augment the image using the abovementioned transformations"""
    transform = T.Compose([
        T.ToTensor(),
        T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomRotation(degrees=5),
        T.RandomHorizontalFlip(p=0.5),
        T.CenterCrop((320, 640)),
        T.Resize((160, 320)),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2)
    ])

    img = cv2.imread('data/sample.png')
    img = transform(img)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = img * 255
    img = img.astype(np.uint8)
    return img

img_input = cv2.imread('data/sample.png')
img_aug = task2(img_input)
cv2.imwrite('data/sample_augmented.png', img_aug)

# EOF
