#!/usr/bin/env python3

import numpy as np
import argparse
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import mnist
from skimage.transform import resize
from tqdm import tqdm

# from diffusion_model import ModelGetter, LossGetter, Forwarder, Backwarder, Scheduler, MODELS_DIR, device

# scale an array of images to a new size


def scale_images(images, new_shape):
    images_list = list()
    for image in tqdm(images):
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

# calculate frechet inception distance


def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Score of Diffusion Model.")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples to use for FID computation.")
    args = parser.parse_args()

    print("Model Loading...")
    model = InceptionV3(include_top=False, pooling='avg',
                        input_shape=(299, 299, 3))
    print("Model Loaded.")

    print("Loading MNIST...")
    (mnist_train, _), (mnist_test, _) = mnist.load_data()
    mnist_train = mnist_train[np.random.choice(
        range(mnist_train.shape[0]), args.n_samples), :]
    mnist_test = mnist_test[np.random.choice(
        range(mnist_test.shape[0]), args.n_samples), :]
    print("MNIST Loaded.")

    print("Preprocessing MNIST...")
    # convert integer to floating point values
    mnist_train = mnist_train.astype('float32')
    mnist_test = mnist_test.astype('float32')

    # resize images
    mnist_train = scale_images(mnist_train, (299, 299, 3))
    mnist_test = scale_images(mnist_test, (299, 299, 3))

    # pre-process images
    mnist_train = preprocess_input(mnist_train)
    mnist_test = preprocess_input(mnist_test)
    print("MNIST Preprocessed.")

    print("Computing FID...")
    # fid between mnist_train and mnist_train
    fid_traintrain = calculate_fid(model, mnist_train, mnist_train)
    print('FID (train vs. train): %.3f' % fid_traintrain)

    # fid between mnist_train and mnist_test
    fid_traintest = calculate_fid(model, mnist_train, mnist_test)
    print('FID (train vs. test): %.3f' % fid_traintest)
