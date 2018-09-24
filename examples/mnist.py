import argparse
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.utils as vutils
from swae.distributions import rand_cirlce2d, rand_ring2d, rand_uniform2d
from swae.models.mnist import MNISTAutoencoder
from swae.trainer import SWAEBatchTrainer
from torchvision import datasets, transforms


def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch MNIST Example')
    parser.add_argument('--datadir', default='/input/', help='path to dataset')
    parser.add_argument('--outdir', default='/output/', help='directory to output images and model checkpoints')
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 500)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='A',
                        help='RMSprop alpha/rho (default: 0.9)')
    parser.add_argument('--distribution', type=str, default='circle', metavar='DIST',
                        help='Latent Distribution (default: circle)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 8)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches to log training status (default: 10)')
    args = parser.parse_args()
    # create output directory
    imagesdir = os.path.join(args.outdir, 'images')
    chkptdir = os.path.join(args.outdir, 'models')
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(imagesdir, exist_ok=True)
    os.makedirs(chkptdir, exist_ok=True)
    # determine device and device dep. args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {'num_workers': args.num_workers, 'pin_memory': False}
    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    # log args
    print('batch size {}\nepochs {}\nRMSprop lr {} alpha {}\ndistribution {}\nusing device {}\nseed set to {}'.format(
        args.batch_size, args.epochs, args.lr, args.alpha, args.distribution, device.type, args.seed
    ))
    # build train and test set data loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.datadir, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.datadir, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=64, shuffle=False, **dataloader_kwargs)
    # create encoder and decoder
    model = MNISTAutoencoder().to(device)
    print(model)
    # create optimizer
    # matching default Keras args for RMSprop
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha)
    # determine latent distribution
    if args.distribution == 'circle':
        distribution_fn = rand_cirlce2d
    elif args.distribution == 'ring':
        distribution_fn = rand_ring2d
    else:
        distribution_fn = rand_uniform2d
    # create batch sliced_wasserstein autoencoder trainer
    trainer = SWAEBatchTrainer(model, optimizer, distribution_fn, device=device)
    # put networks in training mode
    model.train()
    # train networks for n epochs
    print('training...')
    for epoch in range(args.epochs):
        if epoch > 10:
            trainer.weight *= 1.1
        # train autoencoder on train dataset
        for batch_idx, (x, y) in enumerate(train_loader, start=0):
            batch = trainer.train_on_batch(x)
            if (batch_idx + 1) % args.log_interval == 0:
                print('Train Epoch: {} ({:.2f}%) [{}/{}]\tLoss: {:.6f}'.format(
                        epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                        (batch_idx + 1), len(train_loader),
                        batch['loss'].item()))
        # evaluate autoencoder on test dataset
        test_encode, test_targets, test_loss = list(), list(), 0.0
        with torch.no_grad():
            for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
                test_evals = trainer.test_on_batch(x_test)
                test_encode.append(test_evals['encode'].detach())
                test_loss += test_evals['loss'].item()
                test_targets.append(y_test)
        test_encode, test_targets = torch.cat(test_encode).cpu().numpy(), torch.cat(test_targets).cpu().numpy()
        test_loss /= len(test_loader)
        print('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}'.format(
                epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                test_loss))
        print('{{"metric": "loss", "value": {}}}'.format(test_loss))
        # save model
        torch.save(model.state_dict(), '{}/mnist_epoch_{}.pth'.format(chkptdir, epoch + 1))
        # save encoded samples plot
        plt.figure(figsize=(10, 10))
        plt.scatter(test_encode[:, 0], -test_encode[:, 1], c=(10 * test_targets), cmap=plt.cm.Spectral)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Test Latent Space\nLoss: {:.5f}'.format(test_loss))
        plt.savefig('{}/test_latent_epoch_{}.png'.format(imagesdir, epoch + 1))
        plt.close()
        # save sample input and reconstruction
        vutils.save_image(x, '{}/test_samples_epoch_{}.png'.format(imagesdir, epoch + 1))
        vutils.save_image(batch['decode'].detach(),
                          '{}/test_reconstructions_epoch_{}.png'.format(imagesdir, epoch + 1),
                          normalize=True)


if __name__ == '__main__':
    main()
