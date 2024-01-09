import argparse

import mlx.core as mx
import numpy as np

import model
import utils


def generate(
    model_fname,
    num_latent_dims,
    num_img_channels,
    max_num_filters,
    outfile,
    vis_lat_space,
):
    # Load the model
    vae = model.CVAE(num_latent_dims, num_img_channels, max_num_filters)
    vae.load(model_fname)
    print(f"Loaded model with {num_latent_dims} latent dims from {model_fname}")

    # set model to eval mode
    vae.eval()

    # Generate a batch of random latent vectors.
    num_samples = 128

    # During training we have made sure that the distribution in latent
    # space remains close to a normal distribution, so we can sample
    # from a normal distribution to generate new images.

    if vis_lat_space:
        # Generate a grid of 2D latent vectors in a given range and
        # generate images for each of them.
        X, Y = np.meshgrid(np.linspace(-2, 2, 16), np.linspace(-2, 2, 16))
        z = mx.array(np.vstack((X.flatten(), Y.flatten())).T)
        num_rows = 16
    else:
        # Generate a batch of random latent vectors.
        z = mx.random.normal([num_samples, num_latent_dims])
        num_rows = 8

    # Generate images from the latent vectors via the decoder
    images = vae.decode(z)

    # Save all images in a single file
    grid_image = utils.gen_grid_image_from_batch(images, num_rows=num_rows)
    grid_image.save(outfile)
    print(f"Saved {num_samples} generated images to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU acceleration",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--model", type=str, required=True, help="Model filename *.pth")
    parser.add_argument(
        "--latent_dims",
        type=int,
        required=True,
        help="Number of latent dimensions (positive integer)",
    )
    parser.add_argument(
        "--max_filters",
        type=int,
        default=64,
        help="Maximum number of filters in the convolutional layers",
    )

    parser.add_argument(
        "--outfile",
        type=str,
        default="samples.png",
        help="Output filename for the generated samples, e.g. samples.png",
    )
    parser.add_argument(
        "--nimg_channels",
        type=int,
        default=1,
        help="Number of image channels (1 for grayscale, 3 for RGB)",
    )
    parser.add_argument(
        "--vis_lat_space",
        action="store_true",
        help="Visualize the latent space by generating images for a grid of 2D latent vectors. Requires --latent_dims=2",
    )

    args = parser.parse_args()

    # seed
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # set device
    if args.cpu:
        mx.set_default_device(mx.cpu)

    # create output folder if it does not exist
    utils.ensure_folder_exists(args.outfile)

    # Enforce latent_dims=2 for latent space visualization
    if args.vis_lat_space and args.latent_dims != 2:
        raise ValueError(
            "Latent space visualization requires --latent_dims=2. Please set --latent_dims=2 and use a corresponding model."
        )

    generate(
        args.model,
        args.latent_dims,
        args.nimg_channels,
        args.max_filters,
        args.outfile,
        args.vis_lat_space,
    )
