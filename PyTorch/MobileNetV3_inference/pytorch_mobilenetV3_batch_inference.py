#!/usr/bin/env python3


"""
    Usage: python3 PyTorch/MobileNetV3_inference/pytorch_mobilenetV3_batch_inference.py
                    -i ~/PycharmProjects/inference/images/
                    -d GPU
"""

# import required libraries
import sys
import os
import time
from datetime import datetime
import numpy as np
from torchvision import models
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
import argparse
import torch

sys.path.append(os.getcwd())

from utils import list_images
from dataset import MyDataLoader

np.random.seed(1453)  # TODO: Check if PyTorch has anything special for random seed setting
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

print(f"{device} is set for device!")

print("[INFO] MobileNetV3 model is loading ..")
model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(device)
model.eval()

print("[INFO] loading image folder/file ...")
images_path = list(list_images(args['images']))

batch_size = 32

dataloader = MyDataLoader(images_path, batch_size)

experiment_time = str(datetime.now())
experiment_folder = f"experiments/PyTorch_MobileNetV3_{args['device']}_{batch_size}_{experiment_time}"
if not os.path.exists(f"{experiment_folder}"):
    os.mkdir(f"{experiment_folder}")

processing_times = []

# TODO: Grad-Cam results are also going to be in this folder
with torch.no_grad():
    with open(f"{experiment_folder}/inference_results.txt", "w") as f:
        f.write("image_path, predicted_label\n")
        # TODO: I should record inference time for each batch
        for batch, paths in dataloader:
            timer_start = time.process_time()
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
                # print(f"Image path: {paths[i]}, Predicted label: {predicted_labels[i]}")
                f.write(f"{paths[i]}, {predicted_labels[i]}\n")
            timer_stop = time.process_time()
            print(f"Batch process time is {round(timer_stop-timer_start, 2)} secs")
            processing_times.append(round(timer_stop-timer_start, 2))

# TODO: Write script configuration into a yaml file??? and save it to experiment folder
