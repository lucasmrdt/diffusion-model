import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
