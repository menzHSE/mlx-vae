# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import os
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

import utils


# from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py
def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)

    return x


class Encoder(nn.Module):
    """
    A convolutional variational encoder. We do not map the input image
    deterministically to a latent vector. Instead, we map the input to
    a probability distribution in latent space and sample a latent vector
    fron that distribution. In this example, we linearly map the input
    image to a mean vector and a vector of standard deviations that
    parameterize a normal distribution.

    We can then sample from this distribution to generate a new image. Also,
    we can add an auxiliary loss to the network that forces the distribution
    to be close to a standard normal distribution. We use the KL divergence
    between the two distributions as this auxiliary loss.
    """

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters

        # we assume B x #img_channels x 64 x 64 input
        # Todo: add input shape attribute to the model to make it more flexible

        # C x H x W
        img_input_shape = (num_img_channels, 64, 64)

        # layers (with max_num_filters=128)

        num_filters_1 = max_num_filters // 4
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters

        # print(f"Encoder: ")
        # print(f"  num_filters_1={num_filters_1}")
        # print(f"  num_filters_2={num_filters_2}")
        # print(f"  num_filters_3={num_filters_3}")

        # Output: num_filters_1 x 32 x 32
        self.conv1 = nn.Conv2d(num_img_channels, num_filters_1, 3, stride=2, padding=1)
        # Output: num_filters_2 x 16 x 16
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, 3, stride=2, padding=1)
        # Output: num_filters_3 x 8 x 8
        self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, 3, stride=2, padding=1)

        # Shortcuts
        self.shortcut2 = nn.Conv2d(num_filters_1, num_filters_2, 1, stride=2, padding=0)
        self.shortcut3 = nn.Conv2d(num_filters_2, num_filters_3, 1, stride=2, padding=0)

        # Batch Normalizations
        self.bn1 = nn.BatchNorm(num_filters_1)
        self.bn2 = nn.BatchNorm(num_filters_2)
        self.bn3 = nn.BatchNorm(num_filters_3)

        # linear mappings to mean and standard deviation

        # std-dev is not directly outputted but rather as a
        # vector of log-variances. This is because the
        # standard deviation must be positive and the exp()
        # in forward ensures this. It might also be numerically
        # more stable.

        # divide the last two dimensions by 8 because of the 3 strided convolutions
        output_shape = [num_filters_3] + [
            dimension // 8 for dimension in img_input_shape[1:]
        ]

        flattened_dim = math.prod(output_shape)

        self.proj_mu = nn.Linear(flattened_dim, num_latent_dims)
        self.proj_log_var = nn.Linear(flattened_dim, num_latent_dims)

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3(x)))
        x = mx.flatten(x, 1)  # flatten all dimensions except batch

        mu = self.proj_mu(x)
        logvar = self.proj_log_var(x)
        sigma = mx.exp(logvar * 0.5)  # Ensure this is the std deviation, not variance

        # Generate a tensor of random values from a normal distribution
        eps = mx.random.normal(sigma.shape)

        # Perform the reparametrization step ...
        # This allows us to backpropagate through it, which we could not do,
        # if we had just sampled from a normal distribution with mean mu and
        # standard deviation sigma. The multiplication with sigma and addition
        # of mu is just a linear transformation of the random values from the
        # normal distribution. The result is a random value from the distribution
        # with mean mu and standard deviation sigma. Backpropagation is possible
        # because the gradients of the random values are just 1 and the gradients
        # of the linear transformation are just the weights of the linear transformation.
        #z = eps.mul(sigma).add_(mu)
        z = eps * sigma + mu

        # compute KL divergence
        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        self.kl_div = -0.5 * mx.sum(1 + logvar - mu*mu - logvar.exp())
        return z  # return latent vector


class Decoder(nn.Module):
    """A convolutional decoder"""

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.input_shape = None

        # decoder layers
        num_filters_1 = max_num_filters
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters // 4

        # print(f"Decoder: ")
        # print(f"  num_filters_1={num_filters_1}")
        # print(f"  num_filters_2={num_filters_2}")
        # print(f"  num_filters_3={num_filters_3}")

        # C x H x W
        img_output_shape = (num_img_channels, 64, 64)

        # divide the last two dimensions by 8 because of the 3 strided convolutions
        self.input_shape = [num_filters_1] + [
            dimension // 8 for dimension in img_output_shape[1:]
        ]
        flattened_dim = math.prod(self.input_shape)

        # Output: flattened_dim
        self.lin1 = nn.Linear(num_latent_dims, flattened_dim)
        # Output: num_filters_2 x 16 x 16 (with upsample_nearest)
        self.conv1 = nn.Conv2d(
            num_filters_1, num_filters_2, 3, stride=1, padding=1
        )
        # Output: num_filters_1 x 32 x 32 (with upsample_nearest)
        self.conv2 = nn.Conv2d(
            num_filters_2, num_filters_3, 3, stride=1, padding=1
        )
        # Output: #img_channels x 64 x 64 (with upsample_nearest)
        self.conv3 = nn.Conv2d(
            num_filters_3, num_img_channels, 3, stride=1, padding=1
        )

        # Batch Normalizations
        self.bn1 = nn.BatchNorm(num_filters_2)
        self.bn2 = nn.BatchNorm(num_filters_3)

    def __call__(self, z):
        # unflatten the latent vector
        x = self.lin1(z)
        x = x.reshape(-1, self.input_shape[-2], self.input_shape[-1], self.max_num_filters)

        # approximate transposed convolutions with nearest neighbor upsampling      
        x = nn.relu(self.bn1(self.conv1(upsample_nearest(x))))
        x = nn.relu(self.bn2(self.conv2(upsample_nearest(x))))
        x = mx.sigmoid(self.conv3(upsample_nearest(x)))  # sigmoid to ensure pixel values are in [0,1]
        return x


class CVAE(nn.Module):
    """A convolutional Variational Autoencoder"""

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.encoder = Encoder(num_latent_dims, num_img_channels, max_num_filters)
        self.decoder = Decoder(num_latent_dims, num_img_channels, max_num_filters)
        self.kl_div = 0

    # forward pass of the data "x"
    def __call__(self, x):
        z = self.encode(x)
        x = self.decode(z)
        self.kl_div = self.encoder.kl_div
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.trainable_parameters()))
        return nparams

    def save(self, fname):
        # Extract the directory path from the file name
        dir_path = os.path.dirname(fname)

        # Check if the directory path is not empty
        if dir_path:
            # Check if the directory exists, and create it if it does not
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

        # save the model weights
        self.save_weights(fname)

    def load(self, fname):
        self.load_weights(fname, strict=False)
