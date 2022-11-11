#!/usr/bin/env python3

from diffusion_model import ModelGetter, LossGetter, OptimizerGetter, Forwarder, Scheduler, get_mnist_dataset, MODELS_DIR, device
from tqdm.auto import tqdm, trange
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import json
import time
from hashlib import sha1


def one_step(sch, fwd, model, loss_fn, X, label):
    batch_size = X.shape[0]
    t = torch.randint(1, sch.n_steps+1, (batch_size, 1))

    X, label, t = X.to(device), label.to(device), t.to(device)

    x_noisy, noise = fwd.forward(X, t)  # noise image using random time step
    model_pred = model(x_noisy, t, label)  # predict image using model

    return loss_fn(t, x_noisy, noise, model_pred)  # return loss


def one_step_eval(sch, fwd, model, loss_fn, X, label):
    return one_step(sch, fwd, model, loss_fn, X, label).item()


def one_step_training(sch, fwd, opt, model, loss_fn, X, label):
    opt.zero_grad()
    loss = one_step(sch, fwd, model, loss_fn, X, label)
    loss.backward()
    opt.step()
    return loss.item()


def save_model(model, loss, args):
    metadata = json.load(open(os.path.join(MODELS_DIR, "metadata.json")))
    model_id = sha1(str(vars(args)).encode()).hexdigest()
    metadata[model_id] = {
        "args": vars(args),
        "loss": loss,
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    json.dump(metadata, open(os.path.join(
        MODELS_DIR, "metadata.json"), "w"), indent=2)
    state = {"model_state_dict": model.state_dict()}
    torch.save(state, os.path.join(MODELS_DIR, f"{model_id}.pt"))


def int_tuple(s):
    return tuple(int(i) for i in s.split(","))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Model.")
    parser.add_argument("--normalize_range", type=int_tuple,
                        default=(-1, 1), help="Range to normalize the data to.")
    parser.add_argument("--loss", choices=LossGetter.valid_choices,
                        default=LossGetter.default, help="Loss function to use.")
    parser.add_argument("--model", choices=ModelGetter.valid_choices,
                        default=ModelGetter.default, help="Model to use.")
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Number of noising steps.")
    parser.add_argument("--scheduler", choices=Scheduler.valid_choices,
                        default=Scheduler.default, help="Scheduler to use.")
    parser.add_argument("--max_beta", type=int, default=0.999,
                        help="Maximum beta value to use.")
    parser.add_argument("--channels", type=int_tuple, default=(32, 64, 128),
                        help="Channels to use in the model.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Optimizer to use for training.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="The batch size to use for training.")
    args = parser.parse_args()

    print("Dataset loading...")
    dataset = get_mnist_dataset(args.normalize_range, args.batch_size)
    train_loader, test_loader = dataset
    print("Dataset Loaded.")

    print("Building model...")
    sch = Scheduler(args.n_steps, args.scheduler, max_beta=args.max_beta)

    Loss = LossGetter.get_loss(args.loss)
    loss_fn = Loss(sch)

    fwd = Forwarder(sch)

    Model = ModelGetter.get_model(args.model)
    model = Model(sch, fwd, chs=args.channels).to(device)
    model = nn.DataParallel(model).cuda()

    Optimizer = OptimizerGetter.get_optimizer(args.optimizer)
    opt = Optimizer(model.parameters(), lr=args.lr)
    print("Model built.")

    print("Start Training :")
    best_loss = np.inf
    with trange(args.epochs, desc="Epoch", unit="epoch", ascii=True) as te:
        for epoch in te:
            model.train()
            with tqdm(train_loader, desc="Training", unit="batch", leave=False, ascii=True) as bt:
                for (X, label) in bt:
                    loss = one_step_training(
                        sch, fwd, opt, model, loss_fn, X, label)
                    bt.set_postfix(loss=loss)

            model.eval()
            losses = []
            with tqdm(test_loader, desc="Evaluation", leave=False, ascii=True) as be:
                for (inputs, label) in be:
                    loss = one_step_eval(
                        sch, fwd, model, loss_fn, inputs, label)
                    losses.append(loss)
                    be.set_postfix(loss_eval=sum(losses)/len(losses))

            loss_eval = sum(losses)/len(losses)
            if loss_eval <= best_loss:
                best_loss = loss_eval
                save_model(model, best_loss, args)
            te.set_postfix(loss_eval=loss_eval, best_loss=best_loss)
