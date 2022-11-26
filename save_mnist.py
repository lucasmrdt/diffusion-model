#!/usr/bin/env python3

import torchvision
import argparse
from tqdm import trange


from diffusion_model import get_mnist_dataset, MODELS_DIR, device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save MNIST Dataset.")
    parser.add_argument("--n_samples", type=int, default=1024,
                        help="Number of samples to generate.")
    args = parser.parse_args()

    print("Dataset loading...")
    dataset = get_mnist_dataset(batch_size=1)
    train_loader, test_loader = dataset
    print("Dataset Loaded.")

    print("Saving...")
    for i in trange(args.n_samples):
        x, y = next(iter(train_loader))
        torchvision.utils.save_image(x[0, 0, :, :], f"datasets/{i}.png")
