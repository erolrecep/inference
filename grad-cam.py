import numpy as np
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision import transforms
from PIL import Image
import cv2


model = resnet50(weights=ResNet50_Weights.DEFAULT)
target_layers = [model.layer4[-1]]

IMAGE_PATH = "/home/xuser/PycharmProjects/inference/images/awot6aes00291.jpg"

image = Image.open(IMAGE_PATH)
image_cv2 = cv2.imread(IMAGE_PATH)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

image_transformed = transform(image)  # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

input_tensor = torch.unsqueeze(image_transformed, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

targets = [ClassifierOutputTarget(281)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(image_transformed, grayscale_cam, use_rgb=True)
