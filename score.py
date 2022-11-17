#!/usr/bin/env python3

import argparse
import json
import os
from pytorch_fid.fid_score import calculate_fid_given_paths

from diffusion_model import Backwarder, device, MODELS_DIR
from generate import generate


def compute_score(args):
    generate(argparse.Namespace(
        model_id=args.model_id,
        sigma=args.sigma,
        n_samples=2048,
        grid=False
    ))
    score = calculate_fid_given_paths(
        ["samples", "datasets"], 50, device, 2048, num_workers=args.num_workers)
    print(f"FID score: {score}")
    metadata = json.load(open(os.path.join(MODELS_DIR, "metadata.json")))
    metadata[args.model_id] = {
        **metadata[args.model_id],
        "score": score,
    }
    json.dump(metadata, open(os.path.join(
        MODELS_DIR, "metadata.json"), "w"), indent=2)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Score Diffusion Model.")
    parser.add_argument("--model_id", type=str, default="f6e20eee8a2702b174acd04bf0b7fd62b59eb135",
                        help="Model ID to use for generation.")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of workers to use for FID computation.")
    parser.add_argument("--sigma", choices=Backwarder.sigma_valid_choices,
                        default=Backwarder.sigma_default, help="Sigma to use for generation.")
    args = parser.parse_args()

    compute_score(args)
