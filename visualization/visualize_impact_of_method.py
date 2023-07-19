"""
Evaluation results are stored in files of the form:
evaluation_results_{method}_{confidence_threshold}_{max_attempts}_{exp_hash}.txt

They contain a line with the confusion matrix, followed by the accuracy, followed by the average number of attempted
viewpoints per class, followed by the average number of attempted viewpoints for all objects in the test set.

We want to visualize the difference between the methods, for each confidence threshold and max attempts, in terms of
the accuracy and the average number of attempted viewpoints.
"""

import os
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    # Path to the directory containing the evaluation results
    results_dir = Path('/home/andrei/PycharmProjects/nbv_mevs/results/evtest')

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

        # Average number of attempted viewpoints for all objects in the test set is prefixed by 'Average number of
        # attempted viewpoints for all objects in test set: ', not sure on which line it is
        try:
            average_number_of_attempts_for_all_objects = float(
                [line for line in lines if
                 line.startswith('Average number of attempted viewpoints for all objects in test set:')][0].split(' ')[-1])
        except IndexError:
            print(f'Could not find average number of attempted viewpoints for all objects in test set in file {file}')
            continue

        # Store the results
        results[(method, confidence_threshold, max_attempts)] = {
            'accuracy': accuracy,
            'average_number_of_attempts_for_all_objects': average_number_of_attempts_for_all_objects
        }

    # Print them nicely
    for (method, confidence_threshold, max_attempts), result in results.items():
        print(f'{method}, {confidence_threshold}, {max_attempts}:')
        print(f'\tAccuracy: {result["accuracy"]}')
        print(f'\tAverage number of attempted viewpoints for all objects in test set: '
              f'{result["average_number_of_attempts_for_all_objects"]}')

    # Plot the results
    # Plot the accuracy
    fig, ax = plt.subplots()
    # Group the results by method
    results_by_method = {}
    for (method, confidence_threshold, max_attempts), result in results.items():
        if method not in results_by_method:
            results_by_method[method] = []
        results_by_method[method].append({
            'confidence_threshold': confidence_threshold,
            'max_attempts': max_attempts,
            'accuracy': result['accuracy'],
            'average_number_of_attempts_for_all_objects': result['average_number_of_attempts_for_all_objects']
        })

    results_by_max_attempts = {}
    for (method, confidence_threshold, max_attempts), result in results.items():
        if max_attempts not in results_by_max_attempts:
            results_by_max_attempts[max_attempts] = []
        results_by_max_attempts[max_attempts].append({
            'method': method,
            'confidence_threshold': confidence_threshold,
            'accuracy': result['accuracy'],
            'average_number_of_attempts_for_all_objects': result['average_number_of_attempts_for_all_objects']
        })

    # Plot the accuracy
    for method, results_in_method in results_by_method.items():
        ax.scatter([float(result['confidence_threshold']) + (-0.01 if method == 'random' else 0.01) for result in results_in_method],
                   [result['accuracy'] for result in results_in_method],
                   label=method)
        # Add a label showing the max attempts, next to each point
        for result in results_in_method:
            ax.annotate(result['max_attempts'],
                        (float(result['confidence_threshold']), result['accuracy']),
                        xytext=(-10, 0) if method == 'random' else (10, 0), textcoords='offset points')

    # Parse results by max attempts to draw line between points of each method, which have the same max attempts
    # Do not show it in the legend
    max_attempts_list = list(results_by_max_attempts.keys())
    methods = list(results_by_method.keys())
    confidence_threshold_list = list(results_by_method[methods[0]][0].keys())
    for max_attempts in max_attempts_list:
        for confidence_threshold in confidence_threshold_list:
            for method in methods:
                try:
                    print(results[(method, confidence_threshold, max_attempts)])
                    ax.plot(results[(method, confidence_threshold, max_attempts)]['accuracy'],
                            confidence_threshold,
                            label='_nolegend_')
                except KeyError:
                    continue
    ax.set_xlabel('Confidence threshold')
    ax.set_ylabel('Accuracy')
    ax.set_xticks([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
    # set top limit to 1.0
    ax.set_ylim([0.7, 1.0])
    ax.legend()
    plt.savefig('../assets/comparisons/accuracy_vs_confidence_threshold.png')
    plt.show()

    # TODO: the following is not working yet, probably because of some list reference issues

    # Plot the average number of attempted viewpoints for all objects in the test set
    for method, results in results_by_method.items():
        ax.scatter([float(result['confidence_threshold']) for result in results],
                   [result['average_number_of_attempts_for_all_objects'] for result in results],
                   label=method)
        # Add a label showing the max attempts, next to each point
        for result in results:
            ax.annotate(result['max_attempts'],
                        (float(result['confidence_threshold']), result['average_number_of_attempts_for_all_objects']),
                        xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Confidence threshold')
    ax.set_ylabel('Average number of attempted viewpoints for all objects in test set')
    ax.set_xticks(np.arange(0.1, 1.1, 0.1))
    ax.legend()


if __name__ == '__main__':
    main()
