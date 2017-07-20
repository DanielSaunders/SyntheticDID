#!/usr/bin/env python3
import argparse
import sys
import traceback

if sys.version_info < (3, 0):
    sys.stdout.write("Python 2 is not supported. Please use Python 3\n")
    sys.exit(1)

from document import Document
from multiprocessing import Pool

def check_output_count(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("There must be a positive number of images generated")

    return value

def check_level(value):
    value = int(value)
    if value < 1 or value > 5:
        raise argparse.ArgumentTypeError("Level values must be between 1 and 5")

    return value

    print("Usage: python script_generator.py output_num stain_level(1-5) text_noisy_level(1-5)")

def generate_single_image(iter):
    print("Generating image #{}".format(iter + 1))

    # 99585
    try:
        document = Document(args.stain_level, args.text_noise_level)

        document.create()
        document.save(base_dir="/data/synthetic_trial_results/")
        document.save_ground_truth(base_dir="/data/synthetic_trial_results/")

    except Exception as e:
        print(document.random_seed)
        traceback.print_tb(e.__traceback__)

parser = argparse.ArgumentParser(description='Generate some images.')
parser.add_argument('output_count', metavar='N', type=check_output_count, nargs='?', default=5,
                    help='number of images to generate')
parser.add_argument('stain_level', metavar='S', type=check_level, nargs='?', default=1,
                    help='amount of noise in stains')
parser.add_argument('text_noise_level', metavar='T', type=check_level, nargs='?', default=1,
                    help='amount of noise in text')
parser.add_argument('--output_dir', metavar='DIR', default=None,
                    help='directory where final images are to be saved')

args = parser.parse_args()

print("Generating {} images".format(args.output_count))

pool = Pool()

pool.map(generate_single_image, range(args.output_count))


