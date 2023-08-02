"""
Evaluation results are stored in files of the form:
evaluation_results_{method}_{confidence_threshold}_{max_attempts}_{exp_hash}.txt
Method can only 'oracle'

They contain a line with the confusion matrix, followed by the accuracy, followed by the average number of attempted
viewpoints per class, followed by the average number of attempted viewpoints for all objects in the test set.

For this we create a LaTeX table with the accuracy for each confidence threshold and max attempts.
"""

import os
import re
from pathlib import Path

results_dir = Path(__file__).parent.parent

def main(
        results_dir: Path
):
    # List the files matching the pattern
    files = [file for file in os.listdir(results_dir) if re.match(r'evaluation_results_.*\.txt', file)]

    print(f'\n% Parsing {len(files)} files from {results_dir}')

    # Dictionary to store the results
    results = {}

    # Read the results from the files
    for file in files:
        # Parse the file name
        method, confidence_threshold, max_attempts, exp_hash = re.match(
            r'evaluation_results_(.*)_(.*)_(.*)_(.*).txt', file).groups()
        # Read the file
        with open(results_dir / file, 'r') as f:
            lines = f.readlines()
        # Accuracy is prefixed by 'Accuracy: ', not sure on which line it is
        try:
            accuracy = float([line for line in lines if line.startswith('Accuracy: ')][0].split(' ')[1])
        except IndexError:
            print(f'Could not find accuracy in file {file}')
            continue

        # Store the results
        results[(method, confidence_threshold, max_attempts)] = {
            'accuracy': accuracy,
        }

        confidence_thresholds = []
        max_attempts_list = []

        # Print them nicely
        for (method, confidence_threshold, max_attempts), result in results.items():
            if confidence_threshold not in confidence_thresholds:
                confidence_thresholds.append(confidence_threshold)
            if max_attempts not in max_attempts_list:
                max_attempts_list.append(max_attempts)
            # print(f'{method}, {confidence_threshold}, {max_attempts}:')
            # print(f'\tAccuracy: {result["accuracy"]}')
            # print(f'\tAverage number of attempted viewpoints for all objects in test set')

        # Sort the confidence thresholds and max attempts
        confidence_thresholds.sort()
        max_attempts_list.sort()

        # Header is same for both, it's just the confidence thresholds
        print('\\begin{tabular}{|c|', end='')
        for _ in confidence_thresholds:
            print('r|', end='')
        print('} \\hline')
        print(f'Confidence threshold & ', end='')
        for confidence_threshold in confidence_thresholds:
            print(f'           {confidence_threshold} & ', end='')
        print('\\\\ \\hline')
        # Create the LaTeX table
        for max_attempts in max_attempts_list:
            method = 'oracle'
            print(f'{max_attempts} ({method}) \t\t\t & ', end='')
            for confidence_threshold in confidence_thresholds:
                try:
                    if method == 'pcd' and results[('pcd', confidence_threshold, max_attempts)]['accuracy'] > \
                            results[('random', confidence_threshold, max_attempts)]['accuracy']:
                        print(
                            f'\\textbf{{{results[(method, confidence_threshold, max_attempts)]["accuracy"] * 100:.2f}}} & ',
                            end='')
                    elif method == 'random' and results[('pcd', confidence_threshold, max_attempts)][
                        'accuracy'] < \
                            results[('random', confidence_threshold, max_attempts)]['accuracy']:
                        print(
                            f'\\textbf{{{results[(method, confidence_threshold, max_attempts)]["accuracy"] * 100:.2f}}} & ',
                            end='')
                    else:
                        print(
                            f'         {results[(method, confidence_threshold, max_attempts)]["accuracy"] * 100:.2f} & ',
                            end='')
                except KeyError:
                    print('         - & ', end='')
            # delete the last 2 characters, which are '& '
            print('\b\b', end='')
            print('\\\\ \\hline' if method == 'pcd' else '\\\\')

        print('\\end{tabular}')

if __name__ == '__main__':
    gridsearch_results_dir = [
        Path('/home/andrei/PycharmProjects/nbv_mevs/results/oracle_results/evaluation_results_47ca4ba8-resnet34_depth_modelnet10_40views/oracle'),
        Path('/home/andrei/PycharmProjects/nbv_mevs/results/oracle_results/evaluation_results_a1ea9451-resnet18_depth_modelnet10_40views/oracle'),
        Path('/home/andrei/PycharmProjects/nbv_mevs/results/oracle_results/evaluation_results_ac348c87-resnet18_image_modelnet10_40views/oracle')
    ]

    for results_dir in gridsearch_results_dir:
        print(f'\\subsection{{{results_dir.parent.name}}}====================')
        main(results_dir)