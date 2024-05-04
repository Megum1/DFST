import os
import json
import argparse
from loguru import logger

from train import *


def main(args):
    # Load the config
    config_filepath = os.path.join('configs', f'{args.attack}.json')
    with open(config_filepath, 'r') as f:
        config = json.load(f)

    # Create a directory for the model
    model_dir = os.path.join(f'model_dir_{args.attack}')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize the logger
    logger_id = logger.add(
        f"{model_dir}/training.log",
        format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
        level="DEBUG",
    )

    # Train the model
    DEVICE = torch.device(f'cuda:{args.gpu}')
    if args.attack == 'clean':
        train(config, model_dir, logger, DEVICE)
    else:
        poison(config, model_dir, logger, DEVICE)

    # Evaluate the model
    test(config, model_dir, DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--attack', default='dfst', help='attack type')
    args = parser.parse_args()

    main(args)
