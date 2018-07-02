import numpy as np
import torch
import torch.nn.functional as F

from .distributions import rand_cirlce2d


def rand_projections(embedding_dim, num_samples=50):
    """This fn generates `L` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimension size
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor
    """
    theta = [w / np.sqrt((w**2).sum()) for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor)


def _sliced_wasserstein_distance(encoded_samples, distribution_samples, num_projections=50, p=2):
    """Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): embedded training tensor samples
            distribution_samples (torch.Tensor): distribution training tensor samples
            num_projections (int): number of projectsion to approximate sliced wasserstein distance
            p (int): power of distance metric

        Return:
            torch.Tensor
    """
    # derive latent space dimension size from random samples drawn from a distribution in it
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections)
    # calculate projection of the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projection of the random distribution samples
    distribution_projections = distribution_samples.matmul(projections.transpose(0, 1))
    # calculate the sliced wasserstein distance by
    # sorting the samples per projection and
    # calculating the difference between the
    # encoded samples and drawn samples per projection
    wasserstein_distance = torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
    # distance between them (L2 by default for Wasserstein-2)
    wasserstein_distance_p = torch.pow(wasserstein_distance, p)
    # approximate wasserstein_distance for each projection
    return wasserstein_distance_p.mean()


def sliced_wasserstein_distance(encoded_samples, distribution_fn=rand_cirlce2d, num_projections=50, p=2):
    """Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): embedded training tensor samples
            distribution_fn (callable): callable to draw random samples
            num_projections (int): number of projectsion to approximate sliced wasserstein distance
            p (int): power of distance metric

        Return:
            torch.Tensor
    """
    # derive batch size from encoded samples
    batch_size = encoded_samples.size(0)
    # draw samples from latent space prior distribution
    z = distribution_fn(batch_size)
    # approximate wasserstein_distance between encoded and prior distributions
    # for average over each projection
    swd = _sliced_wasserstein_distance(encoded_samples, z, num_projections, p)
    return swd


class SWAEBatchTrainer:
    """Sliced Wasserstein Autoencoder Batch Trainer.

        Args:
            autoencoder (torch.nn.Module): module which implements autoencoder framework
            optimizer (torch.optim.Optimizer): torch optimizer
            distribution_fn (callable): callable to draw random samples
            num_projections (int): number of projectsion to approximate sliced wasserstein distance
            p (int): power of distance metric
            weight_swd (float): weight of divergence metric compared to reconstruction in loss
            device (torch.Device): torch device
    """
    def __init__(self, autoencoder, optimizer, distribution_fn,
                 num_projections=50, p=2, weight_swd=10.0, device=None):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.embedding_dim_ = self.model_ .encoder.embedding_dim_
        self.num_projections_ = num_projections
        self.p_ = p
        self.weight_swd = weight_swd
        self._device = device if device else torch.device('cpu')

    def __call__(self, x):
        return self.eval_on_batch(x)

    def train_on_batch(self, x):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x)
        # backpropagate loss
        evals['loss'].backward()
        # update encoder and decoder parameters
        self.optimizer.step()
        return evals

    def test_on_batch(self, x):
        # reset gradients
        self.optimizer.zero_grad()
        # autoencoder forward pass and loss
        evals = self.eval_on_batch(x)
        return evals

    def eval_on_batch(self, x):
        x = x.to(self._device)
        recon_x, z = self.model_(x)
        bce = F.binary_cross_entropy(recon_x, x)
        l1 = F.l1_loss(recon_x, x)
        w2 = float(self.weight_swd) * sliced_wasserstein_distance(z, self._distribution_fn, self.num_projections_, self.p_)
        loss = bce + l1 + w2
        return {'loss': loss, 'bce': bce, 'l1': l1, 'w2': w2, 'encode': z, 'decode': recon_x}
