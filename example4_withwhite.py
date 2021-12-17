"""
Example 4. Finding camera parameters.
"""
import os
import argparse
import glob

import cv2
import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio

import neural_renderer as nr

from load_off import load_off

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref=None):
        super(Model, self).__init__()

        self.distance_threshold = 6.0
        self.distance_penalty_weight = 1000.0

        # Load mesh vertices and faces
        vertices, faces = load_off(filename_obj)

        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy((imread(filename_ref).max(-1) != 0).astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # camera parameters
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')

        distance = torch.norm(self.camera_position)
        distance_penalty = torch.relu(self.distance_threshold - distance)
        print("distance and distance_penalty: ", distance, distance_penalty)

        loss = torch.sum((self.image_ref[None, :] - image) ** 2) * (1 + distance_penalty)
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()


def make_reference_image(filename_ref, filename_obj):
    model = Model(filename_obj)
    model.cuda()

    model.renderer.eye = nr.get_points_from_angles(2.732, 30, -15)
    images, _, _ = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0]
    imsave(filename_ref, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'ModelNet10/toilet/train/toilet_0001.off'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'gaussian_reference.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example4_result.mp4'))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.make_reference_image:
        make_reference_image(args.filename_ref, args.filename_obj)

    loop = tqdm.tqdm(range(1000))

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()

    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))

        # print(images.shape)

        # image = (images.detach().cpu().numpy()[0].copy() * 255).astype(np.uint8)

        image = (images.detach().cpu().numpy()[0].transpose(1, 2, 0).copy() * 255).astype(np.uint8)

        # https://www.programcreek.com/python/example/89325/cv2.Sobel
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        edges = cv2.addWeighted(np.uint8(abs_sobel_x), 0.5, np.uint8(abs_sobel_y), 0.5, 0)
        # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # print("shapes", image.shape, edges.shape)
        concat = cv2.hconcat([image, edges])
        cv2.putText(concat, f"loss: {loss.item():.2f}", (6, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        imsave('/tmp/_tmp_%04d.png' % i, concat)
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        if loss.item() < 70:
            break
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
