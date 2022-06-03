import os
import argparse

import torch
import numpy as np
import tqdm

import neural_renderer as nr
from old_scripts.find_highest_entropy import edge_detection_loss

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    texture_size = 2
    random_view_seed = 42
    num_views = 100

    np.random.seed(random_view_seed)

    # load .obj
    vertices, faces = nr.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # to gpu

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at')

    entropy_list = []

    # draw object
    loop = tqdm.tqdm(range(0, num_views))
    # writer = imageio.get_writer(args.filename_output, mode='I')

    mean_entropy = 0

    for num in loop:
        loop.set_description('Drawing (mean entropy: %.3f): ' % mean_entropy)
        azimuth = np.random.uniform(0, 360)
        elevation = np.random.uniform(-90, 90)  # TODO: not really random on a sphere!
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images[0]  # [image_size, image_size, RGB]

        entropy = edge_detection_loss(image[np.newaxis, ...]).cpu().detach().numpy().mean()
        # print("entropy.mean()", entropy.mean())
        mean_entropy = entropy

        entropy_list.append(entropy)
        # writer.append_data((255*image).astype(np.uint8))
    mean_entropy = np.mean(entropy_list)
    print('Mean entropy: {}'.format(mean_entropy))

if __name__ == '__main__':
    main()
