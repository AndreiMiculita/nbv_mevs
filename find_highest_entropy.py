"""
Finding highest entropy locally
"""
import argparse
import glob
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import tqdm
from skimage.io import imread, imsave
from torch.special import entr

import canny_filter
import neural_renderer as nr

from load_off import load_off

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

# TODO: https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510/3
# see also
# * https://github.com/pytorch/pytorch/issues/9993
# * https://pytorch.org/docs/master/special.html


def entropy_loss(residual):
    # src https://github.com/pytorch/pytorch/issues/15829#issuecomment-843182236
    # residual.shape[0] -> batch size
    # residual.shape[1] -> Number of channels
    # residual.shape[2] -> Width
    # residual.shape[3] -> Height
    print(residual.shape)
    entropy = torch.zeros(residual.shape[0], 1, requires_grad=True).cuda()
    image_size = float(residual.shape[1] * residual.shape[2] * residual.shape[3])
    for i in range(0, residual.shape[0]):  # loop over batch
        _, counts = torch.unique(residual[i].data.flatten(start_dim=0), dim=0,
                                 return_counts=True)  # flatt tensor and compute the unique values
        p = counts / image_size  # compute the
        entropy[i] = torch.sum(p * torch.log2(p)) * -1
    return entropy.mean()


def other_entropy(input_tensor):
    # src: https://github.com/pytorch/pytorch/issues/15829#issuecomment-725347711
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return entropy


def yet_another_entropy(p, dim = -1, keepdim=None):
    # src: https://github.com/pytorch/pytorch/issues/9993
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)  # can be a scalar, when PyTorch.supports it


def edge_detection_loss(input_tensor):
    return canny_filter.CannyFilter().forward(img=input_tensor)


class Model(nn.Module):
    def __init__(self, filename_obj):
        super(Model, self).__init__()
        # load .obj
        mesh_path = "data/ModelNet10/sofa/train/sofa_0001.off"

        vertices, faces = load_off(mesh_path)

        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # camera parameters
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([6, 10, -14], dtype=np.float32)))

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.eye = self.camera_position
        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        entropy = entr(image).mean() * 10.0
        loss = 1.0 / entropy
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'find_entr_result.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

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
        images, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = (images.detach().cpu().numpy()[0].transpose(1,2,0).copy() * 255).astype(np.uint8)
        edges = cv2.cvtColor(cv2.Canny(image, 30, 150), cv2.COLOR_GRAY2BGR)
        concat = cv2.hconcat([image, edges])
        cv2.putText(concat, f"loss: {loss.item():.2f}", (6, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        imsave('/tmp/_tmp_%04d.png' % i, concat)
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        if loss.item() < 70:
            break
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
