"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import imageio

import neural_renderer as nr
from visualization.plotting_utils import set_axes_equal, show_3d_axes_rgb

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../neural_renderer/examples/data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    elevation = 30
    texture_size = 2

    # load .obj
    vertices, faces = nr.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # to gpu

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at')

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    writer = imageio.get_writer(args.filename_output, mode='I')
    coords = []
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        eye = (eye[0], eye[1], 2.0*eye[2])
        renderer.eye = eye

        print(f"\nangle: {renderer.eye}\n")
        coords.append(renderer.eye)
        images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        writer.append_data((255*image).astype(np.uint8))
    coords_np = np.array(coords)
    np.set_printoptions(suppress=True)
    print(coords_np)

    ax = plt.axes(projection='3d')
    ax.scatter3D(*coords_np.transpose())
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    show_3d_axes_rgb(ax)
    plt.show()
    writer.close()


if __name__ == '__main__':
    main()
