import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

# Read all columns from csv
df = pd.read_csv('../data/entropy-dataset3/entropy_dataset.csv')
print(df)
# Group by the columns "label" and "obj_ind"
df_grouped = df.groupby(['label', 'obj_ind'])

print(df_grouped)

# Print each group
for name, group in df_grouped:
    print(name)
    print(group)

    # Normalize the entropy, divide by the maximum entropy
    group['entropy'] = group['entropy'] / 5.46

    print(group)

    normalize = colors.Normalize(vmin=0, vmax=1)
    # Cmap is jet, with red for 1 and blue for 0
    cmap = plt.cm.get_cmap('jet')

    # Plot the centroids
    scatter = plt.scatter(group['rot_y'], group['rot_x'], c=group['entropy'], cmap=cmap, norm=normalize)

    # Add axis labels
    plt.xlabel('phi')
    plt.ylabel('theta')

    # equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Add legend for the colorbar, with ticks for every 0.1
    sm = plt.colorbar(scatter, ticks=np.arange(0, 1.1, 0.25), fraction=0.022, pad=0.04)

    # Create assets folder if it doesn't exist
    if not os.path.exists('../assets/fibonacci_sphere_vertices/'):
        os.makedirs('../assets/fibonacci_sphere_vertices/')
    plt.savefig('../assets/fibonacci_sphere_vertices/fibonacci_sphere_with_some_entropies.pdf', bbox_inches='tight')

    plt.show()
