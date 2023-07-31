import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from scipy.spatial import Voronoi


# Source: https://stackoverflow.com/a/20678647/13200217
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


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
    group['entropy'] = group['entropy'] / group['entropy'].max()

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
