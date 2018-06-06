import numpy as np
import torch
import torch.nn.functional as F

from .distributions import rand_cirlce2d


def rand_projections(latent_size, num_samples=50):
    """This fn generates `L` random samples from the latent space's unit sphere.

        Args:
            latent_dim (int): latent dimension size
            num_samples (int): number of random projection samples
    """
    theta = [w / np.sqrt((w**2).sum()) for w in np.random.normal(size=(num_samples, latent_size))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor)


def _sliced_wasserstein_distance(encoded_samples, distribution_samples, num_projections=50, p=2):
    # derive latent space dimension size from random samples drawn from a distribution in it
    latent_size = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(latent_size, num_projections)
    # calculate projection of the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projection of the random distribution samples
    distribution_projections = distribution_samples.matmul(projections.transpose(0, 1))
    # calculate the sliced wasserstein distance by sorting the projections
    sorted_differences = torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
    # and computing the distance between them (L2 by default for Wasserstein-2)
    wasserstein_distance = torch.pow(sorted_differences, p)
    # approximate wasserstein_distance for each projection
    return wasserstein_distance.mean()


def sliced_wasserstein_distance(encoded_samples, distribution_fn=rand_cirlce2d, num_projections=50, p=2):
    # derive batch size from encoded samples
    batch_size = encoded_samples.size(0)
    # draw samples from latent space prior distribution
    z = distribution_fn(batch_size)
    # approximate wasserstein_distance between encoded and prior distributions for each projection
    wasserstein_distance = _sliced_wasserstein_distance(encoded_samples, z, num_projections, p)
    # mean wasserstein_distance for all projections
    return wasserstein_distance


class SWAEBatchTrainer:
    """Sliced Wasserstein Autoencoder Batch Trainer."""
    def __init__(self, autoencoder, optimizer, distribution_fn,
                 num_projections=50, p=2, weight_swd=10.0, device='cpu'):
        self.model_ = autoencoder
        self.optimizer = optimizer
        self._distribution_fn = distribution_fn
        self.latent_size_ = self.model_ .encoder.en_dim_
        self.num_projections_ = num_projections
        self.p_ = p
        self.weight_swd = weight_swd
        self._device = device

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
