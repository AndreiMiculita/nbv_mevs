# Script for taking the input image and classifying it with the ResNet-18 model

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import cv2

import viewpoint_utils

# ModelNet10 classes
class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

# Path to the mesh
mesh_path = 'data/teapot.obj'

confidence = 0
confidence_threshold = 0.5

# Choose a random viewpoint
theta = random.random() * np.pi
phi = random.random() * 2 * np.pi - np.pi

attempted_viewpoints = []

while confidence < confidence_threshold:
    # Render the mesh from a random viewpoint
    image = viewpoint_utils.render_mesh(mesh_path, theta, phi)

    # Prepare the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = transforms.ToTensor()(image)

    # Load model from ckpt file "resnet18-5c106cde.pth" and set to eval mode
    model = torchvision.models.resnet18(pretrained=False)
    model.load_state_dict(torch.load('resnet18-5c106cde.pth'))
    model.eval()

    # Get the predictions
    predictions = model(image[None, ...])
    confidence = torch.max(predictions, dim=1).values

    if confidence < confidence_threshold:
        attempted_viewpoints.append([theta, phi])
        # Choose a new viewpoint
        theta, phi = viewpoint_utils.get_new_viewpoint(mesh_path, attempted_viewpoints)
    else:
        print(f'Predicted class: {class_names[torch.argmax(predictions, dim=1)]}')
        print(f'Confidence: {confidence}, Viewpoint coordinates: ({theta}, {phi})')
        print(f'Attempted {len(attempted_viewpoints)} viewpoints')
