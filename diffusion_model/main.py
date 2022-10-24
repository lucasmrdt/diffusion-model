from .logger import console
from .constants import device, SEED
from .utils import seed


def main():
    seed()

    console.log(f'random seed: {SEED}, device: {device}')

    # todo
