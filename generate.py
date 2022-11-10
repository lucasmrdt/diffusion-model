import torch
import torchvision
import numpy as np
import os
from tqdm import trange
import argparse


from model import Glow


def load_model(model):
    state = torch.load(os.path.join('checkpoint', 'bestmodel.pth'))
    model.load_state_dict(state['model_state_dict'])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=256,
                        help="The batch size to use for training.")
    parser.add_argument("--num_channels", type=int, default=16,
                        help="Number of channels to use in the model.")
    parser.add_argument("--num_features", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=5,
                        help="Depth of the model.")
    parser.add_argument("--batchnorm", default=False, action='store_false')
    parser.add_argument("--actnorm", default=True, action='store_true')
    parser.add_argument("--activation", type=str, default='relu')
    parser.add_argument("--n_samples", type=int, default=1024,
                        help="Number of generated samples.")

    args = parser.parse_args()

    # Data Pipeline
    print('Dataset loading...')
    x_dim = (1, 32, 32)
    args.num_levels = int(np.log2(x_dim[1]))-1
    print('Dataset Loaded.')

    print('Model Loading...')
    # Model Pipeline
    model = Glow(in_channels=1,
                 num_channels=args.num_channels,
                 num_levels=args.num_levels,
                 num_steps=args.num_steps,
                 params=args).cuda()
    model = load_model(model)
    model.eval()
    # model = DataParallel(model).cuda()
    print('Model loaded.')

    print('Start Generating :')
    os.makedirs('samples', exist_ok=True)
    with trange(1024, desc="Generated", unit="img") as te:
        for idx in te:
            sample = torch.randn(1,
                                 x_dim[1]*x_dim[2],
                                 1,
                                 1).cuda()
            x, _ = model(sample, None, True)
            x = x[:, :, 2:30, 2:30]
            torchvision.utils.save_image(
                x, os.path.join('samples', f'{idx}.png'))
