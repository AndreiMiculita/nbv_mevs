"""
We load the ResNet ckpts and the dataset of rendered images, and we evaluate the accuracy of the model on the dataset.
There are 40 views per object; we want to know also if there are objects for which a majority of the views are
misclassified as a different class.
Also we want to know what happens if we sum the features of the 40 views and then softmax the result.
We can't use a dataloader, we have to laod the images one by one, because the dataloader does not preserve the object
id.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import tqdm

from PIL import Image

# ckpts are one dir up in data/ckpt_files
ckpt_dir = Path(__file__).parent.parent / 'data/per_object_evaluation/ckpt_files'

image_dataset_dirs = [

]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']


def main():
    for image_dataset_dir in image_dataset_dirs:
        for ckpt_file in ckpt_dir.iterdir():
            if 'depth' in ckpt_file.name:
                continue
            # Write to a log file with the ckpt file name
            with open(f'evaluate_resnet_per_object_{ckpt_file.stem}_{image_dataset_dir.name}.log', 'w') as f:
                sys.stdout = f
                sys.stderr = f
                print('dead')
                # Load the pretrained ResNet-18 model
                resnet = torchvision.models.resnet18(
                    pretrained=False) if 'resnet18' in ckpt_file.name else torchvision.models.resnet34(pretrained=False)
                # Replace the last layer with a linear layer with 10 outputs (one for each class)
                resnet.fc = nn.Linear(512, 10)
                # Load the checkpoint
                resnet.load_state_dict(torch.load(ckpt_file))
                # Set the model to evaluation mode
                resnet.eval()

                # Define the test transforms
                test_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ])

                # First we retrieve the test set filenames from modelnet10_test.txt
                with open(image_dataset_dir / 'modelnet10_test.txt', 'r') as f:
                    test_set_filenames = f.readlines()

                test_set_filenames = [filename.strip() for filename in test_set_filenames]

                # We group the filenames by class + object id; the filenames are of the form
                # {class_name}_{object_id}_theta_{theta}_phi_{phi}_vc_{view_counter}.png
                # Note that theta and phi and vc are in the filename
                # Also discard the extension
                test_set_filenames_grouped = {}
                for filename in test_set_filenames:
                    filename_temp = filename.replace('night_stand', 'nightstand')
                    print(len(filename_temp.split('_')))
                    class_name, object_id, _, theta, _, phi, _, vc = filename_temp.split('_')
                    # Recover original class & filename
                    class_name = class_name.replace('nightstand', 'night_stand')
                    filename = filename + '.png'
                    if (class_name, object_id) not in test_set_filenames_grouped:
                        test_set_filenames_grouped[(class_name, object_id)] = []
                    test_set_filenames_grouped[(class_name, object_id)].append(
                        (filename, float(theta), float(phi), int(vc)))

                # We now have a dict of the form {(class_name, object_id): [(filename, theta, phi, vc), ...], ...}, we build a
                # batch of 40 images for each object, and we evaluate the model on the batch
                correct = 0
                total = 0
                for (class_name, object_id), filenames in tqdm.tqdm(test_set_filenames_grouped.items()):
                    correct_per_object = 0
                    total_per_object = 0
                    print(f'Testing object {object_id} of class {class_name}')
                    # Sort the filenames by theta, then by phi, then by vc
                    filenames.sort(key=lambda x: (x[1], x[2], x[3]))
                    # Build the batch
                    batch = []
                    for filename, _, _, _ in filenames:
                        image = Image.open(image_dataset_dir / class_name / filename)
                        image = test_transforms(image)
                        batch.append(image)
                    batch = torch.stack(batch)
                    # Evaluate the model on the batch
                    # We check if most of the views are classified as the correct class
                    # Otherwise we print that most views are another class
                    with torch.no_grad():
                        outputs = resnet(batch)
                        _, predicted = torch.max(outputs.data, 1)
                        total += len(predicted)
                        correct += (predicted == torch.tensor(
                            [class_names.index(class_name)] * len(predicted))).sum().item()

                        correct_per_object += (
                                predicted == torch.tensor([class_names.index(class_name)] * len(predicted))).sum().item()
                        total_per_object += len(predicted)

                        print((predicted == torch.tensor([class_names.index(class_name)] * len(predicted))).sum().item(),
                              len(predicted) / 2)

                        # If most of the views are classified as the correct class
                        if (predicted == torch.tensor([class_names.index(class_name)] * len(predicted))).sum().item() > len(
                                predicted) / 2:
                            print(f'\tMajority is correct')
                            print(
                                f'\tCorrect: {correct_per_object}, Total: {total_per_object}, Accuracy: {100 * correct_per_object / total_per_object:.2f}%')
                            print(f'\tCorrect overall: {correct}, Total: {total}, Accuracy: {100 * correct / total:.2f}%\n')
                        else:
                            print(f'\tMajority is incorrect!')
                            print(
                                f'\tCorrect: {correct_per_object}, Total: {total_per_object}, Accuracy: {100 * correct_per_object / total_per_object:.2f}%\n')
                            if class_names[predicted[0]] != class_name:
                                print(f'\tClassified as {class_names[predicted[0]]} instead of {class_name}')
                            else:
                                print(f'\tClassified as {class_names[predicted[0]]} but not by majority')
                            # Print probabilities as a dict, one line per class
                            print('\tProbabilities:')
                            print('\t{')
                            for i, class_name in enumerate(class_names):
                                print(f'\t\t{class_name}: {outputs[0][i].item()},')
                            print('\t}\n')
                            print(f'\tCorrect overall: {correct}, Total: {total}, Accuracy: {100 * correct / total:.2f}%\n')

                print(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')


if __name__ == '__main__':
    main()
