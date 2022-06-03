import torch
from torch import nn as nn


def entropy_loss(residual):
    # src https://github.com/pytorch/pytorch/issues/15829#issuecomment-843182236
    # residual.shape[0] -> batch size
    # residual.shape[1] -> Number of channels
    # residual.shape[2] -> Width
    # residual.shape[3] -> Height
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