# Script for taking the input image and classifying it with the ResNet-18 model

import os

import cv2
import torch
import torchvision
import torchvision.transforms as transforms


resnet_path = '../data/ckpt_files/resnet18_modelnet10.ckpt'
image_path = '../data/image-dataset/bathtub/'


def main():
    # ModelNet10 classes
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    # Load model from ckpt file with load_from_checkpoint and set to eval mode
    # Note that our model has 10 classes, so we need to change the last layer to have 10 outputs
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(torch.load(resnet_path))
    model.eval()

    images_list = os.listdir(image_path)
    images_list.sort()

    total = len(images_list)
    sofar = 0
    correct = 0

    for image in images_list:
        print(f'Image: {image}')
        # Retrieve the image
        image = cv2.imread(image_path + image)

        # Prepare the image
        image = cv2.resize(image, (224, 224))
        image = transforms.ToTensor()(image)

        # Get the predictions
        predictions = model(image[None, ...])
        confidence = torch.max(predictions, dim=1).values

        predicted_class = class_names[torch.argmax(predictions, dim=1)]

        print(f'Predicted class: {predicted_class}')
        print(f'Confidence: {confidence}')
        # Print predictions, with the class names
        for i in range(len(class_names)):
            print(f'{class_names[i]}: {predictions[0][i]}')

        if predicted_class == 'bathtub':
            correct += 1

        sofar += 1

        print(f'Accuracy so far: {correct / sofar}')

    print(f'Accuracy: {correct / total}')


if __name__ == '__main__':
    main()