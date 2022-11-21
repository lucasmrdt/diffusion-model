#!/usr/bin/env python3

from score import compute_score
from diffusion_model import ModelGetter, LossGetter, OptimizerGetter, Forwarder, Scheduler, Backwarder, get_mnist_dataset, MODELS_DIR, device
from hashlib import sha1
from tqdm.auto import tqdm, trange
from functools import partial
import time
import json
import argparse
import os
import optuna
import numpy as np
import torch.nn as nn
import torch
import warnings

warnings.filterwarnings("ignore")


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


def save_model(model_id, model, loss, args):
    state = {"model_state_dict": model.state_dict()}
    torch.save(state, os.path.join(MODELS_DIR, f"{model_id}.pt"))
    metadata = json.load(open(os.path.join(MODELS_DIR, "metadata.json")))
    metadata[model_id] = {
        "args": vars(args),
        "loss": loss,
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    json.dump(metadata, open(os.path.join(
        MODELS_DIR, "metadata.json"), "w"), indent=2)


def int_tuple(s):
    return tuple(int(i) for i in s.split(","))


def train(args):
    if args.verbose:
        print("Args:", args)

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
    model = Model(sch, fwd, chs=args.channels,
                  time_attn=args.time_attn, mid_attn=args.mid_attn, depth=args.depth, width=args.width).to(device)
    model = nn.DataParallel(model).cuda()

    Optimizer = OptimizerGetter.get_optimizer(args.optimizer)
    opt = Optimizer(model.parameters(), lr=args.lr)
    print("Model built.")
    model_id = sha1(str(vars(args)).encode()).hexdigest()

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
                save_model(model_id, model, best_loss, args)
            te.set_postfix(loss_eval=loss_eval, best_loss=best_loss)
    print("Training finished.")
    if args.compute_fid:
        score = compute_score(argparse.Namespace(
            model_id=model_id,
            num_workers=32,
            sigma=args.sigma
        ))
        return score
    else:
        return best_loss


def objective(args, trial):
    normalize_range = trial.suggest_categorical("normalize_range",
                                                [(-1, 1), (0, 1), (-0.5, 0.5)])
    n_steps = trial.suggest_categorical("n_steps",
                                        [1, 2, 5, 10, 20, 50, 100, 500, 1000])
    lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4])
    scheduler = trial.suggest_categorical("scheduler", Scheduler.valid_choices)
    loss = trial.suggest_categorical("loss", LossGetter.valid_choices)
    channels = trial.suggest_categorical("channels",
                                         [(32, 64, 128), (16, 32, 64), (8, 16, 32), (32, 64, 128, 256), (16, 32, 64, 128)])
    model = trial.suggest_categorical("model", ModelGetter.valid_choices)

    train_args = argparse.Namespace(
        # fixed params
        batch_size=args.batch_size,
        max_beta=args.max_beta,
        optimizer=args.optimizer,
        sigma=args.sigma,
        verbose=args.verbose,
        epochs=args.epochs,
        compute_fid=True,

        # variable params
        normalize_range=normalize_range,
        n_steps=n_steps,
        lr=lr,
        scheduler=scheduler,
        loss=loss,
        channels=channels,
        model=model,
    )
    return train(train_args)


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
    parser.add_argument("--depth", type=int, default=4,
                        help="Depth of the model. (only for ConvModel)")
    parser.add_argument("--width", type=int, default=32,
                        help="Width of the model. (only for ConvModel)")
    parser.add_argument("--time_attn", action='store_true', default=False,
                        help="Use time attention.")
    parser.add_argument("--mid_attn", action='store_true', default=False,
                        help="Use mid attention.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Optimizer to use for training.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="The batch size to use for training.")
    parser.add_argument("--optuna", action='store_true', default=False,
                        help="Use optuna for hyperparameter tuning.")
    parser.add_argument("--verbose", action='store_true', default=False,
                        help="Print more information.")
    parser.add_argument("--compute_fid", action='store_true', default=False,
                        help="Compute FID score.")
    parser.add_argument("--sigma", choices=Backwarder.sigma_valid_choices,
                        default=Backwarder.sigma_default, help="Sigma to use for generation.")
    args = parser.parse_args()

    if args.optuna:
        objective = partial(objective, args)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
    else:
        train(args)
