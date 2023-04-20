# Calculates and plots the mean and std dev of the number of points in each point cloud, for each object
# For this, it reads every pcd file
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

data_dir = '/home/andrei/PycharmProjects/Pointnet2_PyTorch/pointnet2/data/modelnet10_andrei_partial'

num_points_list_all = []

print("Reading point clouds...")

for split in ["train", "test"]:
    if split == "train":
        shape_ids = [
            line.rstrip()
            for line in open(
                os.path.join(data_dir, "modelnet10_train.txt")
            )
        ]
    else:
        shape_ids = [
            line.rstrip()
            for line in open(
                os.path.join(data_dir, "modelnet10_test.txt")
            )
        ]

    shape_names = ["_".join(x.split("_")[0:-7]) for x in shape_ids]
    # list of (shape_name, shape_txt_file_path) tuple
    datapath = [
        (
            shape_names[i],
            os.path.join(data_dir, shape_names[i], shape_ids[i])
            + ".pcd",
        )
        for i in range(len(shape_ids))
    ]

    num_points_list = {}
    # For each object, calculate the mean and std dev of the number of points in each point cloud
    for shape_name, shape_txt_file_path in datapath:
        if shape_name not in num_points_list:
            num_points_list[shape_name] = []

        # read pcd file
        pcd = o3d.io.read_point_cloud(shape_txt_file_path)

        # compute the number of points in the point cloud
        num_points = np.asarray(pcd.points).shape[0]
        num_points_list[shape_name].append(num_points)

    num_points_list_all.append(num_points_list)

# Print lists for debugging
print(num_points_list_all)

# Sort the keys so that the order is the same for train and test
keys = sorted(num_points_list_all[0].keys())
fig, ax = plt.subplots()
index = np.arange(len(keys))
bar_width = 0.35
opacity = 0.8

# Now we plot them with matplotlib, with a grouped bar chart with mean and std dev for each object
for idx, num_points_list in enumerate(num_points_list_all):

    # get the mean and std dev of the number of points for each object
    means = [np.mean(num_points_list[key]) for key in keys]
    std_devs = [np.std(num_points_list[key]) for key in keys]

    if idx == 0:
        rects1 = ax.bar(index, means, bar_width,
                        alpha=opacity, capsize=5,
                        yerr=std_devs, label='Train')
    else:
        rects2 = ax.bar(index + bar_width, means, bar_width,
                        alpha=opacity, capsize=5,
                        yerr=std_devs, label='Test')

ax.set_xlabel('Class')
ax.set_ylabel('Number of points')
ax.set_title('Number of points per class')
ax.set_xticks(index + bar_width / 2)
# Display night_stand as nightstand
keys = [key.replace('_', '') for key in keys]
ax.set_xticklabels(keys)
# For the yticks we want to convert 000 to k
ax.set_yticklabels(['{:.0f}k'.format(x / 1000) for x in ax.get_yticks()])
ax.legend()

# Make it wider
fig.set_size_inches(9, 5)
fig.tight_layout()
plt.show()

# Save the figure as pdf in the assets folder
fig.savefig('../assets/num_points_per_class.pdf', bbox_inches='tight')

# Print the mean and std dev of the number of points for all objects
for idx, num_points_list in enumerate(num_points_list_all):
    # replace nightstand back to night_stand
    keys = [key.replace('nightstand', 'night_stand') for key in keys]
    means = [np.mean(num_points_list[key]) for key in keys]
    std_devs = [np.std(num_points_list[key]) for key in keys]
    print("Mean: ", np.mean(means))
    print("Std Dev: ", np.mean(std_devs))
