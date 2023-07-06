# Script for taking the input image and classifying it with the ResNet-18 model

import argparse

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import pipeline.get_captures
import pipeline.get_new_viewpoint
from geometry_utils.convert_coords import as_spherical
from geometry_utils.fibonacci_sphere import fibonacci_sphere


def main(args=None):
    # ModelNet10 classes
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    mesh_path = args.mesh_path
    confidence = 0
    confidence_threshold = args.confidence

    attempted_viewpoints = []

    possible_viewpoints = []
    for point in fibonacci_sphere(10):
        [_, theta, phi] = as_spherical(list(point))
        theta -= np.pi
        theta = -theta
        phi += np.pi
        possible_viewpoints.append((theta, phi))

    prediction_accumulator = np.zeros(len(class_names))

    # Get a random viewpoint
    theta, phi = pipeline.get_new_viewpoint.get_new_viewpoint_coords(mesh_path, attempted_viewpoints,
                                                                     possible_viewpoints,
                                                                     method=args.method)

    print(
        f'Initial viewpoint coordinates: ({theta:.2f}, {phi:.2f}) radians '
        f'({np.degrees(theta):.2f}, {np.degrees(phi):.2f}) degrees')

    # Load model from ckpt file and set to eval mode, set outputs to size 10
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(torch.load(args.classification_model))
    model.eval()

    while confidence < confidence_threshold:
        # Retrieve the image, given the viewpoint
        image = pipeline.get_captures.get_image(mesh_path, theta, phi)

        # Prepare the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = transforms.ToTensor()(image)

        # Get the predictions
        prediction_accumulator += model(image[None, ...]).detach().numpy()[0]
        confidence = np.max(prediction_accumulator)
        attempted_viewpoints.append([theta, phi])

        if confidence < confidence_threshold:
            print(f'Might be: {class_names[np.argmax(prediction_accumulator)]}')
            # Choose a new viewpoint
            print(f'\nConfidence too low ({confidence:.2f}), choosing new viewpoint, method: {args.method}')
            theta, phi = pipeline.get_new_viewpoint.get_new_viewpoint_coords(mesh_path, attempted_viewpoints,
                                                                             possible_viewpoints)
            print(
                f'New viewpoint coordinates: ({theta:.2f}, {phi:.2f}) radians '
                f'({np.degrees(theta):.2f}, {np.degrees(phi):.2f}) degrees')
        else:
            print(f'\nDone! Predicted class: {class_names[np.argmax(prediction_accumulator)]}')
            print(f'Confidence: {confidence:.2f}')
            print(f'Attempted {len(attempted_viewpoints)} viewpoints')
            # Softmax the accumulator
            probabilities = torch.nn.Softmax(dim=0)(torch.tensor(prediction_accumulator))
            accumulator_dict = dict(zip(class_names, zip(prediction_accumulator, probabilities)))
            print('Class      : Accumulated : Probability (softmaxed)')
            print('\n'.join([f'{key.ljust(11)}: {value: 06.2f} : {prob: .3f}' for key, (value, prob) in accumulator_dict.items()]))


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='The pipeline for multi-view 3D object classification')
    parser.add_argument('--mesh_path', default='~/datasets/ModelNet10/bathtub/test/bathtub_0107.off', type=str,
                        help='Path to the mesh file')
    parser.add_argument('--classification_model', type=str,
                        default='../data/ckpt_files/resnet18_modelnet10.ckpt',
                        help='Path to the classification model checkpoint')
    parser.add_argument('--entropy_prediction_model', type=str,
                        default='../data/ckpt_files/06186690-epoch=34-val_loss=0.01.ckpt',
                        help='Path to the entropy prediction model checkpoint (pointnet)')
    parser.add_argument('--differential_rendering_model', type=str)
    parser.add_argument('--confidence', metavar='confidence', type=float, default=20, help='Confidence threshold')
    parser.add_argument('--method', metavar='method', type=str, default='random',
                        help='Method for choosing the next viewpoint, choose from: random, diff, pcd')
    args = parser.parse_args()
    main(args)
