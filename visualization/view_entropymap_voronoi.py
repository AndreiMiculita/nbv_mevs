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

    grid = np.zeros((7, 7))

    voronoi_to_grid_map = {
        (43, 148): (0, 0),
        (119, 177): (0, 1),
        (117, 142): (0, 2),
        (165, 134): (0, 3),
        (212, 149): (0, 4),
        (257, 129): (0, 5),
        (296, 148): (0, 6),
        (27, 116): (1, 0),
        (81, 125): (1, 1),
        (43, 157): (1, 2),
        (43, 158): (1, 3),
        (43, 159): (1, 4),
        (43, 160): (1, 5),
        (43, 161): (1, 6),
        (43, 162): (2, 0),
        (43, 163): (2, 1),
        (43, 164): (2, 2),
        (43, 165): (2, 3),
        (43, 166): (2, 4),
        (43, 167): (2, 5),
        (43, 168): (2, 6),
        (43, 169): (3, 0),
        (43, 170): (3, 1),
        (43, 171): (3, 2),
        (43, 172): (3, 3),
        (43, 173): (3, 4),
        (43, 174): (3, 5),
        (43, 175): (3, 6),
        (43, 176): (4, 0),
        (43, 177): (4, 1),
        (43, 178): (4, 2),
        (43, 179): (4, 3),
        (43, 180): (4, 4),
        (43, 181): (4, 5),
        (43, 182): (4, 6),
        (43, 183): (5, 0),
        (43, 184): (5, 1),
        (43, 185): (5, 2),
        (43, 186): (5, 3),
        (43, 187): (5, 4),
        (43, 188): (5, 5),
        (43, 189): (5, 6),
        (43, 190): (6, 0),
        (43, 191): (6, 1),
        (43, 192): (6, 2),
        (43, 193): (6, 3),
        (43, 194): (6, 4),
        (43, 195): (6, 5),
        (43, 196): (6, 6)
    }

    # Create a Voronoi diagram from the group
    vor = Voronoi(group[['rot_y', 'rot_x']])

    # Plot the Voronoi diagram
    regions, vertices = voronoi_finite_polygons_2d(vor)

    cmap = plt.cm.get_cmap('jet', 10)

    normalize = colors.Normalize(vmin=0, vmax=6)

    # Plot the centroids
    scatter = plt.scatter(group['rot_y'], group['rot_x'], c=cmap(normalize(group['entropy'])))

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

    # Plot on left side of the figure
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Color regions by entropy, using colormap "jet" mapped to entropy, and transparent edges
    for idx, region in enumerate(regions):
        polygon = vertices[region]
        ax1.fill(*zip(*polygon), color=cmap(normalize(group['entropy'].iloc[idx])), edgecolor='none')

    # Plot the centroids, with coordinate labels next to the points
    ax1.plot(group['rot_y'], group['rot_x'], 'ko')
    # Add labels next to each point showing its coordinates rot_x and rot_y
    for (x, y) in zip(group['rot_y'], group['rot_x']):
        ax1.text(x, y, '(' + str(int(x)) + ', ' + str(int(y)) + ')')

    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, 180)

    # populate the grid based on the map
    for i in range(len(group)):
        try:
            print((int(group['rot_x'].iloc[i]), int(group['rot_y'].iloc[i])))
            grid[
                voronoi_to_grid_map[(int(group['rot_x'].iloc[i]), int(group['rot_y'].iloc[i]))][0],
                voronoi_to_grid_map[(int(group['rot_x'].iloc[i]), int(group['rot_y'].iloc[i]))][1]
            ] = group['entropy'].iloc[i]
        except KeyError:
            print(f"KeyError for {str(int(group['rot_x'].iloc[i])), str(int(group['rot_y'].iloc[i]))}")
            continue
        print(f"Key worked for {str(group['rot_x'].iloc[i])}, {str(group['rot_y'].iloc[i])}")

    # plot the grid on the right side of the figure
    ax2.imshow(grid, cmap=cmap, interpolation='nearest', origin='lower', extent=[0, 7, 0, 7])

    plt.show()
