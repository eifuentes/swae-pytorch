import numpy as np
import torch
from sklearn.datasets import make_circles


def rand_cirlce2d(batch_size):
    """This function generates 2D samples from a filled-circle distribution in a 2-dimensional space

        Args:
            batch_size (int): number of batch samples
    """
    r = np.random.uniform(size=(batch_size))
    theta = 2*np.pi*np.random.uniform(size=(batch_size))
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z_ = np.array([x, y]).T
    return torch.from_numpy(z_).type(torch.FloatTensor)


def rand_ring2d(batch_size):
    """This function generates 2D samples from a hollowed-cirlce distribution in a 2-dimensional space

        Args:
            batch_size (int): number of batch samples
    """
    temp = make_circles(2*batch_size, noise=.01)
    z_ = np.squeeze(temp[0][np.argwhere(temp[1] == 0), :])
    return torch.from_numpy(z_).type(torch.FloatTensor)


def rand_uniform2d(batch_size):
    """This function generates 2D samples from a uniform distribution in a 2-dimensional space

        Args:
            batch_size (int): number of batch samples
    """
    z_ = 2*(np.random.uniform(size=(batch_size, 2))-0.5)
    return torch.from_numpy(z_).type(torch.FloatTensor)
