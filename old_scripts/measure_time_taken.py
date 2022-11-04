import numpy as np

from find_highest_entropy_noprints import main
import time
import os
import random


def get_random_file_in_dir(category_dir: str) -> str:
    random_file = random.choice(os.listdir(category_dir))
    full_name = os.path.join(category_dir, random_file)
    return full_name


if __name__ == '__main__':
    categories = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    n_iterations = 10
    times = {}

    last_string = ""

    for category in categories:
        times[category] = []
        for i in range(n_iterations):
            category_dir = f'/home/andrei/datasets/ModelNet10/{category}/train/'
            random_file_in_dir = get_random_file_in_dir(category_dir)
            args = ['--filename_obj', random_file_in_dir]
            start_time = time.time()
            main(args)
            duration = time.time() - start_time
            times[category].append(duration)
        towrite = f"Times for category {category} {times[category]}"
        print(towrite)
        last_string += towrite + "\n"
        towrite = f"Average time for category {category}: {np.mean(times[category])}"
        print(towrite)
        last_string += towrite + "\n"

    print(last_string)