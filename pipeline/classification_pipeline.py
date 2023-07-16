import argparse
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from node_weighted_graph import Node
from pipeline.get_captures import get_capture
from pipeline.get_new_viewpoint import get_new_viewpoint_coords


def main(
        mesh_path: Path,
        classification_model: Path,
        possible_viewpoints_path: Path,
        confidence_threshold: float,
        method: str,
        max_attempts: int = 10,
        pcd_entropy_prediction_model: Path = None,
        differentiable_rendering_model: Path = None
):
    """
    Main function for taking the input image and classifying it based on multiple viewpoints.
    :param mesh_path: the path to the mesh file
    :param classification_model: the path to the classification model
    :param possible_viewpoints_path: the path to the possible viewpoints graph
    :param confidence_threshold: the confidence threshold for the classification model; if the confidence is below this
    threshold, the model will choose a new viewpoint
    :param method: the method for choosing a new viewpoint
    :param pcd_entropy_prediction_model: the path to the point cloud embedding network model
    :param differentiable_rendering_model: the path to the differentiable rendering model
    :return: the class id, confidence and list of attempted viewpoints
    """

    # ModelNet10 classes
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    confidence = 0

    attempted_viewpoints = []

    # Read the graph from the graphml file
    possible_viewpoints_graph_nx = nx.read_graphml(possible_viewpoints_path)

    # Example node: <node id="A: 90.0, 0.0"/>
    possible_viewpoints = []
    for node in possible_viewpoints_graph_nx.nodes(data=True):
        theta, phi = float(node[1]['theta']), float(node[1]['phi'])
        possible_viewpoints.append((theta, phi))

    # Convert to our custom Node class
    possible_viewpoints_graph = []
    for i, (theta, phi) in enumerate(possible_viewpoints):
        node = Node(str(i), np.radians(theta), np.radians(phi), 0)
        possible_viewpoints_graph.append(node)

    # Copy the edges as well, with Node.add_neighbor(); we use the neighbor's name to find it in the graph
    for i, node in enumerate(possible_viewpoints_graph_nx.nodes):
        for neighbor in possible_viewpoints_graph_nx.neighbors(node):
            # Find neighbor's index in graph nx
            neighbor_index = list(possible_viewpoints_graph_nx.nodes).index(neighbor)
            # Add neighbor to the node
            possible_viewpoints_graph[i].add_neighbor(possible_viewpoints_graph[neighbor_index])

    prediction_accumulator = np.zeros(len(class_names))

    # Get a random viewpoint
    theta, phi = get_new_viewpoint_coords(mesh_path, attempted_viewpoints, possible_viewpoints_graph)

    print(
        f'Initial viewpoint coordinates: ({theta:.2f}, {phi:.2f}) radians '
        f'({np.degrees(theta):.2f}, {np.degrees(phi):.2f}) degrees'
    )

    # Load model from ckpt file and set to eval mode, set outputs to size 10
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, len(class_names))
    model.load_state_dict(torch.load(classification_model))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model = model.to(device)

        while confidence < confidence_threshold and len(attempted_viewpoints) < max_attempts:
            print(f'\nAttempting viewpoint: ({theta:.2f}, {phi:.2f}) radians ')
            # Retrieve the image, given the viewpoint
            image = get_capture(mesh_path, theta, phi)

            if image is None:
                print('Could not retrieve image, exiting...')
                exit(1)

            # Prepare the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = transforms.ToTensor()(image).to(device)

            # Get the predictions
            prediction_accumulator += model(image[None, ...]).cpu().numpy()[0]
            confidence = np.max(torch.nn.Softmax(dim=0)(torch.tensor(prediction_accumulator)).cpu().numpy())
            attempted_viewpoints.append((theta, phi))

            if confidence < confidence_threshold:
                print(f'Might be: {class_names[np.argmax(prediction_accumulator)]}')
                # Choose a new viewpoint
                print(f'\nConfidence too low ({confidence:.2f}), choosing new viewpoint, method: {method}')
                theta, phi = get_new_viewpoint_coords(mesh_path, attempted_viewpoints, possible_viewpoints_graph,
                                                      method,
                                                      pcd_model_path=pcd_entropy_prediction_model if method == 'pcd' else None)
                print(
                    f'New viewpoint coordinates: ({theta:.2f}, {phi:.2f}) radians '
                    f'({np.degrees(theta):.2f}, {np.degrees(phi):.2f}) degrees'
                )
            else:
                print(f'\nDone! Predicted class: {class_names[np.argmax(prediction_accumulator)]}')
                print(f'Confidence: {confidence:.5f}')
                print(f'Attempted {len(attempted_viewpoints)} viewpoints')
                # Softmax the accumulator
                probabilities = torch.nn.Softmax(dim=0)(torch.tensor(prediction_accumulator))
                accumulator_dict = dict(zip(class_names, zip(prediction_accumulator, probabilities)))
                print('Class      : Accumulated : Probability (softmaxed)')
                print('\n'.join([f'{key.ljust(11)}:      {value: 06.2f} : {prob:.3f}' for key, (value, prob) in
                                 accumulator_dict.items()]))

    return np.argmax(prediction_accumulator), confidence, attempted_viewpoints


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='The pipeline for multi-view 3D object classification')
    parser.add_argument('--mesh_path', default='~/datasets/ModelNet10/bathtub/test/bathtub_0107.off', type=str,
                        help='Path to the mesh file')
    parser.add_argument('--classification_model', type=str,
                        default='../data/ckpt_files/resnet18_modelnet10.ckpt',
                        help='Path to the classification model checkpoint')
    parser.add_argument('--pcd_entropy_prediction_model', type=str,
                        default='../data/ckpt_files/06186690-epoch=34-val_loss=0.01.ckpt',
                        help='Path to the entropy prediction model checkpoint (pointnet)')
    parser.add_argument('--differentiable_rendering_model', type=str, default='/')  # TODO
    parser.add_argument('--possible_viewpoints_path', type=str, default='../config/entropy_views_10_better.graphml')
    parser.add_argument('--confidence', metavar='confidence', type=float, default=0.99, help='Confidence threshold')
    parser.add_argument('--method', metavar='method', type=str, default='pcd',
                        help='Method for choosing the next viewpoint, choose from: random, diff, pcd')
    args = parser.parse_args()
    main(
        mesh_path=Path(args.mesh_path),
        classification_model=Path(args.classification_model),
        pcd_entropy_prediction_model=Path(args.pcd_entropy_prediction_model),
        differentiable_rendering_model=Path(args.differentiable_rendering_model),
        possible_viewpoints_path=Path(args.possible_viewpoints_path),
        confidence_threshold=args.confidence,
        method=args.method
    )
