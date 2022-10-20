"""
Finding highest entropy locally
"""
import argparse
import glob
import os
from typing import List

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import tqdm
from skimage.io import imread, imsave

import canny_filter
import neural_renderer as nr

from load_off import load_off

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


# TODO: https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510/3
# see also
# * https://github.com/pytorch/pytorch/issues/9993
# * https://pytorch.org/docs/master/special.html


def edge_detection_loss(input_tensor):
    filter = canny_filter.CannyFilter(use_cuda=True)
    # for param in filter.parameters():
    #     param.requires_grad = False
    blurred, grad_x, grad_y, grad_magnitude, grad_orientation = filter.forward(img=input_tensor)
    # print("grad_magnitude.shape", grad_magnitude.shape)
    # imsave('/tmp/grad_magnitude.png', grad_magnitude.data.cpu().numpy()[0][0])
    # exit(0)
    return grad_magnitude


class Model(nn.Module):
    def __init__(self, filename_obj):
        super(Model, self).__init__()
        # Load mesh vertices and faces
        vertices, faces = load_off(filename_obj)

        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # camera parameters
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([3, 5, -7], dtype=np.float32)))

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.eye = self.camera_position
        self.renderer = renderer

        self.entropy = 0

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        self.entropy = edge_detection_loss(image[np.newaxis, ...])
        print("entropy.mean()", self.entropy.mean())
        loss = 10000.0 / self.entropy.mean()
        self.image = image
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()


def main(args: List[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'find_entr_result.mp4'))
    parser.add_argument('-g', '--gpu', type=int, default=0)

    if args is None:
        args = parser.parse_args()  # parse arguments from sys.argv
    else:
        args = parser.parse_args(args)  # parse arguments from given list

    model = Model(args.filename_obj)
    model.cuda()

    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loop = tqdm.tqdm(range(1000))
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        # print(images.shape)
        image = (model.image.data.cpu().numpy() * 255).astype(np.uint8)[0]
        print(image.shape)
        # print(images.detach().cpu().numpy()[0].max(), images.detach().cpu().numpy()[0].min())
        edges = model.entropy.detach().cpu().numpy()[0][0]
        edges = (edges * 255 / edges.max()).astype(np.uint8)
        print("image.max(), image.min()", image.max(), image.min())
        print("edges.max(), edges.min()", edges.max(), edges.min())
        print(edges.shape)
        print("camera position", model.camera_position.detach().cpu().numpy())
        # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        print("shapes", image.shape, edges.shape)
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
