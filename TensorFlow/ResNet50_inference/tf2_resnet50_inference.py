#!/usr/bin/env python3


# TODO: Load all images from the user input image folder
# TODO: Preprocess images to the size of the model input size
# TODO: Write a dataloader to feed the model with the optimized data
# TODO: Load model
# TODO: Do inference
# TODO: Time each and every step of the pipeline
# TODO: Write inference results to a log file with each input image name and it's corresponding timings
# \_ The experiment log file name should be "year_month_day_time.csv"


# import required libraries
import sys
import os
from datetime import datetime

sys.path.append(os.getcwd())
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from utils import list_images
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="images dataset path")
ap.add_argument("-d", "--device", required=True, help="Compute device, either CPU or GPU")
args = vars(ap.parse_args())

if args['device'] == 'CPU':
    tf.config.set_visible_devices([], 'GPU')
    EXPERIMENT_NAME = f"tf2_resnet50_inference_CPU_{str(datetime.now().date())}-{str(datetime.now().time())}.txt"

else:
    EXPERIMENT_NAME = f"tf2_resnet50_inference_GPU_{str(datetime.now().date())}-{str(datetime.now().time())}.txt"

model = ResNet50(weights='imagenet')

# load all images from the provided folder
# TODO: make sure the user input folder/file exists

images_path = list(list_images(args['images']))
print(images_path)
# img_path = 'elephant.jpg'

# TODO: Write a dataloader to load data optimized

all_images = []

with open(EXPERIMENT_NAME, "w") as f:
    for img_path in images_path:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        all_images.append(x)

        preds = model.predict(x)
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        print('Predicted:', decode_predictions(preds, top=5)[0])  # TODO: lean the prediction result with class name
        # and percentage
        f.write(f"{img_path},{decode_predictions(preds, top=5)[0]}\n")

# TODO: Write inference results to a log file with their timings
