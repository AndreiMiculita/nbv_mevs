"""
We run the classification pipeline on the test set and evaluate the results.
We cannot use a dataloader because we need to get the file paths based on the viewpoint.
"""

import hashlib
import time
from pathlib import Path

import numpy as np

from pipeline.classification_pipeline import main as classification_pipeline_main

# This contains the file paths to the test set meshes, it's in the data dir which is one dir up from this file
root_dir = Path(__file__).parent.parent
test_set_file = root_dir / 'data/ModelNet10/test_set_filenames.txt'
results_dir = root_dir / 'results'

method_list = ['random', 'pcd']
confidence_threshold_list = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
max_attempts_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def main():
    # Print current dir
    print(Path.cwd())

    # ModelNet10 classes
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    # Read the test set file
    with open(test_set_file, 'r') as f:
        test_set = f.readlines()
    test_set = [Path(line.strip()) for line in test_set]

    for confidence_threshold in confidence_threshold_list:
        for max_attempts in max_attempts_list:
            for method in method_list:
                # Confusion matrix, on the rows we have the true labels, on the columns we have the predicted labels
                confusion_matrix = np.zeros((len(class_names), len(class_names)))

                # Attempted viewpoints, one list for each class
                attempted_viewpoints_per_class = [[] for _ in range(len(class_names))]

                # There are 10 viewpoints, for each one we have a counter of how many times it was chosen
                viewpoints_counter = {}

                # Experiment hash, to avoid overwriting results
                exp_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[0:8]

                with open(results_dir / f'evaluation_results_'
                                        f'{method}_{confidence_threshold}_{max_attempts}_{exp_hash}.txt', 'w') as f:
                    # Run the classification pipeline on the test set
                    for mesh_path in test_set:
                        label = class_names.index(mesh_path.parent.parent.name)
                        print(f'Testing on mesh: {mesh_path}')
                        class_id, confidence, attempted_viewpoints = classification_pipeline_main(
                            mesh_path=mesh_path,
                            classification_model=root_dir / 'data/ckpt_files/resnet18_modelnet10.ckpt',
                            possible_viewpoints_path=root_dir / 'entropy_views_10.graphml',
                            confidence_threshold=confidence_threshold,
                            method=method,
                            max_attempts=max_attempts,
                            pcd_entropy_prediction_model=root_dir / 'data/ckpt_files/06186690-epoch=34-val_loss=0.01.ckpt'
                        )
                        confusion_matrix[label][class_id] += 1
                        attempted_viewpoints_per_class[class_id].append(attempted_viewpoints)
                        # Write stats to file
                        f.write(f'Mesh: {mesh_path}\n')
                        f.write(f'Label: {label}\n')
                        f.write(f'Class ID: {class_id}\n')
                        f.write(f'Confidence: {confidence}\n')
                        f.write(f'Attempted viewpoints: {attempted_viewpoints}\n')
                        for viewpoint in attempted_viewpoints:
                            viewpoints_counter[viewpoint] = viewpoints_counter.get(viewpoint, 0) + 1

                    # Print the confusion matrix
                    print(confusion_matrix)
                    f.write('Confusion matrix:\n')
                    f.write(str(confusion_matrix))
                    f.write('\n')
                    f.write(f'Accuracy: {np.trace(confusion_matrix) / np.sum(confusion_matrix)}\n')
                    f.write('Average number of attempted viewpoints per class:\n')
                    total_number_of_attempts = 0
                    for i, attempted_viewpoints in enumerate(attempted_viewpoints_per_class):
                        number_of_meshes = len(attempted_viewpoints)
                        number_of_attempts = sum([len(attempted_viewpoints) for attempted_viewpoints in attempted_viewpoints])
                        f.write(f'{class_names[i]}: {number_of_attempts / number_of_meshes}\n')
                        total_number_of_attempts += sum(
                            [len(attempted_viewpoints) for attempted_viewpoints in attempted_viewpoints])
                    f.write('Average number of attempted viewpoints for all objects in test set: ')
                    f.write(f'{total_number_of_attempts / len(test_set)}\n')
                    f.write('Most chosen viewpoints:\n')
                    for i, (viewpoint, count) in enumerate(sorted(viewpoints_counter.items(), key=lambda x: x[1], reverse=True)):
                        f.write(f'{i + 1}. {viewpoint}: {count}\n')


if __name__ == '__main__':
    main()
