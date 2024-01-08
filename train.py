import argparse
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import dataset
import model
import loss
import utils


def loss_fn(model, X):
    X_recon = model(X)
    return loss.mse_kl_loss(X, X_recon, model.kl_div)


def save_model(vae, name, epoch):
    fname = f"{name}_{epoch:03d}.npz"
    utils.ensure_folder_exists(fname)
    vae.save(fname)


def train_epoch(model, tr_iter, loss_and_grad_fn, optimizer, epoch):
    # set model to training mode
    model.train()

    # reset stats
    running_loss = 0.0
    throughput_list = []

    # iterate over training batches
    for batch_count, batch in enumerate(tr_iter):
        X = mx.array(batch["image"])

        throughput_tic = time.perf_counter()

        # forward pass + backward pass + update
        loss, grads = loss_and_grad_fn(model, X)
        optimizer.update(model, grads)
        # Evaluate updated model parameters
        mx.eval(model.parameters(), optimizer.state)

        throughput_toc = time.perf_counter()
        throughput = X.shape[0] / (throughput_toc - throughput_tic)
        throughput_list.append(throughput)
        running_loss += loss

        if batch_count > 0 and (batch_count % 10 == 0):
            print(
                f"Epoch {epoch:4d}: Loss {(running_loss.item() / batch_count):6.5f} | "
                f"Batch {batch_count:5d} | "
                f"Throughput {throughput:10.2f} images/second",
                end="\r",
            )

        batch_count = batch_count + 1

        #### end of loop over training batches ####

    return running_loss, throughput_list, batch_count


def train(batch_size, num_epochs, learning_rate, num_latent_dims, max_num_filters):
    # Load the training and test data
    img_size = (64, 64)
    tr_iter, _, num_img_channels = dataset.mnist(
        batch_size=batch_size, img_size=img_size
    )

    # Load the model
    vae = model.CVAE(num_latent_dims, num_img_channels, max_num_filters)

    # Allocate memory and initialize parameters
    mx.eval(vae.parameters())
    print("Number of trainable params: {:0.04f} M".format(vae.num_params() / 1e6))

    # loss and optimizer
    loss_and_grad_fn = nn.value_and_grad(vae, loss_fn)
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # file name to save model every epoch
    fname_save_every_epoch = f"models/vae_mnist_filters_{vae.max_num_filters:04d}_dims_{vae.num_latent_dims:04d}"

    print("Starting training ...")

    for e in range(num_epochs):
        # reset iterators and stats at the beginning of each epoch
        tr_iter.reset()

        # train one epoch
        tic = time.perf_counter()
        running_loss, throughput_list, batch_count = train_epoch(
            vae, tr_iter, loss_and_grad_fn, optimizer, e
        )
        toc = time.perf_counter()

        # reset iterators
        tr_iter.reset()
        samples_per_sec = mx.mean(mx.array(throughput_list))

        # print stats
        print(
            f"Epoch {e:4d}: Loss {(running_loss.item() / batch_count):6.5f} | "
            f"Throughput {samples_per_sec.item():10.2f} images/second | ",
            f"Time {toc - tic:8.3f} (s)",
        )

        # save model
       
        save_model(vae, fname_save_every_epoch, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a convolutional VAE with mlx.")

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU (cuda/mps) acceleration",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--batchsize", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--max_filters",
        type=int,
        default=128,
        help="Maximum number of filters in the convolutional layers",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion-mnist", "cifar-10", "cifar-100", "celeb-a"],
        default="mnist",
        help="Select the dataset to use (mnist, fashion-mnist, cifar-10, cifar-100, celeb-a)",
    )
    parser.add_argument(
        "--latent_dims",
        type=int,
        required=True,
        help="Number of latent dimensions (positive integer)",
    )

    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print("Options: ")
    print(f"  Device: {'GPU' if not args.cpu else 'CPU'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Max number of filters: {args.max_filters}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of latent dimensions: {args.latent_dims}")

    train(args.batchsize, args.epochs, args.lr, args.latent_dims, args.max_filters)
