"""
Evaluation results are stored in files of the form:
evaluation_results_{method}_{confidence_threshold}_{max_attempts}_{exp_hash}.txt
Method can only be 'random' or 'pcd'.

They contain a line with the confusion matrix, followed by the accuracy, followed by the average number of attempted
viewpoints per class, followed by the average number of attempted viewpoints for all objects in the test set.

We want to visualize the difference between the methods, for each confidence threshold and max attempts, in terms of
the accuracy difference between the methods.
For this we create a LaTeX table with the accuracy difference for each confidence threshold and max attempts.
"""

import os
import re
from pathlib import Path


def main():
    show_difference = False

    # Path to the directory containing the evaluation results
    results_dir = Path('/home/andrei/PycharmProjects/nbv_mevs/results/evaluation_results_10views_image_resnet18')

    # List the files matching the pattern
    files = [file for file in os.listdir(results_dir) if re.match(r'evaluation_results_.*\.txt', file)]

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
        print(f'{method}, {confidence_threshold}, {max_attempts}:')
        print(f'\tAccuracy: {result["accuracy"]}')
        print(f'\tAverage number of attempted viewpoints for all objects in test set')

    # Sort the confidence thresholds and max attempts
    confidence_thresholds.sort()
    max_attempts_list.sort()

    # Create the LaTeX table
    for max_attempts in max_attempts_list:
        if show_difference:
            print(f'{max_attempts} & ', end='')
            for confidence_threshold in confidence_thresholds:
                    difference = results[('pcd', confidence_threshold, max_attempts)]['accuracy'] - \
                                    results[('random', confidence_threshold, max_attempts)]['accuracy']
                    difference *= 100
                    # Print the difference, rounded to 2 decimals; bold if it is positive
                    if difference > 0:
                        print(f'\\textbf{{{difference:.3f}}} & ', end='')
                    else:
                        print(f'       {difference:.3f}  & ', end='')
            print('\\\\ \\hline')
        else:
            for method in ['random', 'pcd']:
                print(f'{max_attempts} ({method}) & ', end='')
                for confidence_threshold in confidence_thresholds:
                    if method == 'pcd' and results[('pcd', confidence_threshold, max_attempts)]['accuracy'] > \
                            results[('random', confidence_threshold, max_attempts)]['accuracy']:
                        print(f'\\textbf{{{results[(method, confidence_threshold, max_attempts)]["accuracy"]*100:.2f}}} & ', end='')
                    elif method == 'random' and results[('pcd', confidence_threshold, max_attempts)]['accuracy'] < \
                            results[('random', confidence_threshold, max_attempts)]['accuracy']:
                        print(f'\\textbf{{{results[(method, confidence_threshold, max_attempts)]["accuracy"]*100:.2f}}} & ', end='')
                    else:
                        print(f'{results[(method, confidence_threshold, max_attempts)]["accuracy"]*100:.2f} & ', end='')
                print('\\\\ \\hline')


if __name__ == '__main__':
    main()
