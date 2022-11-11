#!/usr/bin/env python3

import torch
import numpy as np
import os
import argparse
import json
import time
from hashlib import sha1
from tqdm.auto import tqdm, trange

from diffusion_model import Model, Loss, Forwarder, Optimizer, Scheduler, get_mnist_dataset, MODELS_DIR, device


def save_model(model, loss, args):
    metadata = json.load(open(os.path.join(MODELS_DIR, "metadata.json")))
    model_id = sha1(str(vars(args)).encode()).hexdigest()
    metadata[model_id] = {
        "args": vars(args),
        "loss": loss,
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    json.dump(metadata, open(os.path.join(MODELS_DIR, "metadata.json"), "w"))
    state = {'model_state_dict': model.state_dict()}
    torch.save(state, os.path.join(MODELS_DIR, f"{model_id}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    # parser.add_argument("--optimizer", type=str, default='adam',
    #                     help="Optimizer to use for training.")
    # parser.add_argument("--gamma", type=float, default=0.5,
    #                     help="Gamma value to use for learning rate schedule.")
    # parser.add_argument("--lr", type=float, default=0.001,
    #                     help="The learning rate to use for training.")
    # parser.add_argument("--wd", type=float, default=0,
    #                     help="Weight decay to use for training.")
    # parser.add_argument("--num_channels", type=int, default=16,
    #                     help="Number of channels to use in the model.")
    # parser.add_argument("--num_features", type=int, default=256)
    # parser.add_argument("--batchnorm", default=False, action='store_false')
    # parser.add_argument("--actnorm", default=True, action='store_true')
    # parser.add_argument("--activation", type=str, default='relu')

    # pre-processing parameters
    parser.add_argument("--normalize_range", type=tuple,
                        default=(-1, 1), help="Range to normalize the data to.")

    # loss parameters
    parser.add_argument("--loss", choices=Loss.valid_choices,
                        default=Loss.default, help="Loss function to use.")

    # scheduler parameters
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Number of noising steps.")
    parser.add_argument("--scheduler", choices=Scheduler.valid_choices,
                        default=Scheduler.default, help="Scheduler to use.")
    parser.add_argument("--max_beta", type=int, default=0.999,
                        help="Maximum beta value to use.")

    # model parameters
    parser.add_argument("--channels", type=tuple, default=(32, 64, 128),
                        help="Channels to use in the model.")

    # training parameters
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Optimizer to use for training.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="The batch size to use for training.")

    args = parser.parse_args()

    # Data Pipeline
    print('Dataset loading...')
    dataset = get_mnist_dataset(args.normalize_range, args.batch_size)
    train_loader, test_loader = dataset
    print('Dataset Loaded.')

    print("Building model...")
    sch = Scheduler(args.n_steps, args.scheduler, max_beta=args.max_beta)
    loss_fn = Loss(sch, args.loss)
    fwd = Forwarder(sch)
    model = Model(sch, fwd, args.channels).to(device)
    # model = DataParallel(model).cuda()
    opt = Optimizer(args.optimizer, model.parameters(), lr=args.lr)
    print("Model built.")

    print('Start Training :')
    best_loss = np.inf
    with trange(args.epochs, desc="Epoch", unit="epoch", position=0) as te:
        for epoch in te:
            model.train()
            with tqdm(train_loader, desc="Training", unit="batch", position=1, leave=False) as bt:
                for n_batch, (X, label) in enumerate(bt):
                    loss = model.one_step_training(opt, loss_fn, X, label)
                    bt.set_postfix(loss=loss)

            model.eval()
            losses = []
            with tqdm(test_loader, desc="Evaluation", position=2, leave=False) as be:
                for n_batch, (inputs, label) in enumerate(be):
                    losses.append(model.one_step_eval(loss_fn, inputs, label))
                    be.set_postfix(loss_eval=sum(losses)/len(losses))
            loss_eval = sum(losses)/len(losses)
            if loss_eval <= best_loss:
                best_loss = loss_eval
                save_model(model, best_loss, args)
            te.set_postfix(loss_eval=loss_eval, best_loss=best_loss)
