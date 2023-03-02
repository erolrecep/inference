#!/usr/bin/env python3


# import required libraries
import numpy as np
import torch
import cv2


torch.backends.cudnn.benchmark = True


def make_transform(image):
    # swap the color channels from BGR to RGB, resize it, and scale
    # the pixel values to [0, 1] range
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    # subtract ImageNet mean, divide by ImageNet standard deviation,
    # set "channels first" ordering, and add a batch dimension
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    image -= MEAN
    image /= STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    # return the preprocessed image
    return image


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, all_imgs):
        super(InferenceDataset, self,).__init__()

        self.all_imgs = all_imgs
        self.original_image = None
        self.transform = make_transform(self.original_image)       # some infer transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        self.original_image = self.all_imgs[idx]
        return self.transform
