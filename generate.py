#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import os
import json
import argparse
import numpy as np
from tqdm import trange


from diffusion_model import ModelGetter, LossGetter, Forwarder, Backwarder, Scheduler, MODELS_DIR, device


def get_model_info(model_id=None):
    metadata = json.load(open(os.path.join(MODELS_DIR, "metadata.json")))
    if not model_id in metadata:
        print("No model_id provided, using best model...")
        model_id = max(metadata, key=lambda k: metadata[k]["loss"])
        print(f"Using: {model_id}")
    model_args = metadata[model_id]["args"]
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pt")
    state = torch.load(model_path,  map_location=device)
    return state["model_state_dict"], model_args


def dict_without_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Diffusion Model.")
    parser.add_argument("--model_id", type=str, default="058c3c3806a6b75e6744bc39de97da0ca372abbd",
                        help="Model ID to use for generation.")
    parser.add_argument("--sigma", choices=Backwarder.sigma_valid_choices,
                        default=Backwarder.sigma_default, help="Sigma to use for generation.")
    parser.add_argument("--n_samples", type=int, default=1024,
                        help="Number of samples to generate.")
    parser.add_argument("--grid", action='store_true', default=False,
                        help="Whether to generate a grid of samples.")
    args = parser.parse_args()

    print("Model Loading...")
    model_state, model_args = get_model_info(args.model_id)

    sch = Scheduler(model_args["n_steps"], model_args["scheduler"],
                    max_beta=model_args["max_beta"])

    Loss = LossGetter.get_loss(model_args["loss"])

    fwd = Forwarder(sch)

    Model = ModelGetter.get_model(model_args["model"])
    model = Model(sch, fwd,
                  **dict_without_keys(model_args, ["scheduler"]))
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_state)

    bkw = Backwarder(sch, model,
                     is_predicting_noise=Loss.is_predicting_noise(), sigma=args.sigma)
    print("Model loaded.")

    model.eval()
    print("Start Generating :")
    os.makedirs("samples", exist_ok=True)

    x = bkw.sample(args.n_samples, (32, 32))
    x = x[:, :, 2:30, 2:30]  # remove padding
    if args.grid:
        file_name = f"{args.model_id}.png" if args.model_id else "grid.png"
        torchvision.utils.save_image(x, os.path.join("samples", file_name),
                                     nrow=np.sqrt(args.n_samples).astype(int))
    else:
        for i in trange(args.n_samples, desc="Saving"):
            torchvision.utils.save_image(x[i, 0, :, :], f"samples/{i}.png")
