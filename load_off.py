import numpy as np
import torch


# part of code from https://stackoverflow.com/questions/31129968/off-files-on-python
def load_off(filename_off, normalization=True, texture_size=4, load_texture=False,
             texture_wrapping='REPEAT', use_bilinear=True):
    with open(filename_off) as file:
        if 'OFF' != file.readline().strip():
            raise ('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        vertices = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

        # normalize into a unit cube centered zero
        if normalization:
            vertices -= vertices.min(0)[0][None, :]
            vertices /= torch.abs(vertices).max()
            vertices *= 2
            vertices -= vertices.max(0)[0][None, :] / 2

        return vertices, faces