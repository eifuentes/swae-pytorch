import torch.nn as nn
import torch.nn.functional as F


class MNISTEncoder(nn.Module):
    """MNIST Encoder from Original Paper Implementation."""
    def __init__(self, depth=16, lrelu_slope=0.2, inter_dim=128, en_dim=2):
        super(MNISTEncoder, self).__init__()

        self.depth_ = depth
        self.lrelu_slope_ = lrelu_slope
        self.inter_dim_ = inter_dim
        self.en_dim_ = en_dim

        self.features = nn.Sequential(
            nn.Conv2d(1, self.depth_ * 1, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.depth_ * 1, self.depth_ * 1, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.depth_ * 1, self.depth_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.depth_ * 2, self.depth_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.depth_ * 2, self.depth_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.depth_ * 4, self.depth_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.AvgPool2d(kernel_size=2, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.depth_ * 4 * 4 * 4, self.inter_dim_),
            nn.ReLU(inplace=True),
            nn.Linear(self.inter_dim_, self.en_dim_)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.depth_ * 4 * 4 * 4)
        x = self.fc(x)
        return x


class MNISTDecoder(nn.Module):
    """MNIST Decoder from Original Paper Implementation."""
    def __init__(self, depth=16, lrelu_slope=0.2, inter_dim=128, en_dim=2):
        super(MNISTDecoder, self).__init__()

        self.depth_ = depth
        self.lrelu_slope_ = lrelu_slope
        self.inter_dim_ = inter_dim
        self.en_dim_ = en_dim

        self.fc = nn.Sequential(
            nn.Linear(self.en_dim_, self.inter_dim_),
            nn.Linear(self.inter_dim_, self.depth_ * 4 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.depth_ * 4, self.depth_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.depth_ * 4, self.depth_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.depth_ * 4, self.depth_ * 4, kernel_size=3, padding=0),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.depth_ * 4, self.depth_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.depth_ * 4, self.depth_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.depth_ * 2, self.depth_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.depth_ * 2, 1, kernel_size=3, padding=1)
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 4 * self.depth_, 4, 4)
        z = self.features(z)
        return F.sigmoid(z)


class MNISTAutoencoder(nn.Module):
    """MNIST Autoencoder from Original Paper Implementation."""
    def __init__(self, depth=16, lrelu_slope=0.2, inter_dim=128, en_dim=2):
        super(MNISTAutoencoder, self).__init__()

        self.depth_ = depth
        self.lrelu_slope_ = lrelu_slope
        self.inter_dim_ = inter_dim
        self.en_dim_ = en_dim

        self.encoder = MNISTEncoder(depth, lrelu_slope, inter_dim, en_dim)
        self.decoder = MNISTDecoder(depth, lrelu_slope, inter_dim, en_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
