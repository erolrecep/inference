#!/usr/bin/env python3


"""
    Usage: python3 PyTorch/EfficientNet_inference/pytorch_efficientnet_batch_inference.py
                    -i ~/PycharmProjects/inference/images/
                    -d GPU
"""

# import required libraries
import sys
import os
import numpy as np
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_V2_M_Weights
import argparse
import torch

sys.path.append(os.getcwd())

from utils import list_images
from dataset import MyDataLoader

np.random.seed(1453)
torch.backends.cudnn.benchmark = True

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="images dataset path")
ap.add_argument("-d", "--device", required=True, help="Compute device, either CPU or GPU")
args = vars(ap.parse_args())

if args['device'] == 'GPU':
    # Define the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print("[INFO] EfficientNet-M model is loading ..")
model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT).to(device)
model.eval()

print("[INFO] loading image folder/file ...")
images_path = list(list_images(args['images']))

batch_size = 32

dataloader = MyDataLoader(images_path, batch_size)

with torch.no_grad():
    for batch, paths in dataloader:
        # Move the batch of images to the device
        batch = batch.to(device)

        # Perform inference on the batch
        with torch.no_grad():
            outputs = model(batch)

        # Convert the outputs to probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get the predicted class labels
        predicted_labels = torch.argmax(probabilities, dim=1)

        # Print the predicted labels and the corresponding image paths
        for i in range(len(paths)):
            print(f"Image path: {paths[i]}, Predicted label: {predicted_labels[i]}")
