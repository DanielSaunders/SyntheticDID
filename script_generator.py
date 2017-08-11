#!/usr/bin/env python3
"""
Script Generator generates a set of synthetic handwritten text documents

Using DivaDID as well as opencv, this module will generate an arbitrary number
of images that are intended to have the appearance of authentic, handwritten
text documents.

Alongside the generated images, binary "ground truth" images are generated,
which are the absolute binarized form of the text documents. That means that
the binarized data is exclusively the text itself, and not other noise.
"""
import argparse
import random
import sys
from multiprocessing import Pool
from document import Document
import cv2

# Check to make sure that we are using Python 3
if sys.version_info < (3, 0):
    sys.stdout.write("Python 2 is not supported. Please use Python 3\n")
    sys.exit(1)

DEFAULT_DIR = "/data/synthetic_trial_" + str(random.randint(10000, 100000))


def check_output_count(value):
    """
    Custom `type` function for argparse to check for valid --output-count

    A custom function that verifies that values passed into the program as
    --output-count are valid. Valid means any positive integer.
    """
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("There must be a positive number of"
                                         "images generated")

    return value


def check_level(value):
    """
    Custom `type` function for argparse to check for valid level args

    A custom function that verifies that values passed into the program as
    --stain_level and --text_noise_level are valid. Valid means integers
    between 1 and 5 inclusive.
    """
    value = int(value)
    if value < 1 or value > 5:
        raise argparse.ArgumentTypeError(
            "Level values must be between 1 and 5")

    return value


def generate_single_image(fn_args):
    """
    Generate and save a single image

    Using arguments passed in the command line, generate a single image and
    save it to the given output directory.
    """
    print("Generating image #{}".format(fn_args['iter'] + 1))

    # 99585
    try:
        document = Document(fn_args['args'].stain_level,
                            fn_args['args'].text_noise_level,
                            output_loc=fn_args['args'].output_dir)


        document.create(bypass=fn_args['args'].bypass_divadid)
        document.save()
        document.save_ground_truth()

    except cv2.error as exception:
        print(document.random_seed)
        print(type(exception))
        print(exception.args)
        print(exception.args)

        with open("errors.txt", "a+") as errors:
            errors.write("{}\n".format(document.random_seed))

def main():
    """
    Main entrance point into program

    A simple function that parses arguments as appropriate, prepares needed
    child processes, and generates images.
    """
    parser = argparse.ArgumentParser(description='Generate some images.')
    parser.add_argument('output_count', metavar='N', type=check_output_count,
                        nargs='?', default=8,
                        help='number of images to generate')
    parser.add_argument('stain_level', metavar='S', type=check_level,
                        nargs='?', default=1, help='amount of noise in stains')
    parser.add_argument('text_noise_level', metavar='T', type=check_level,
                        nargs='?', default=1, help='amount of noise in text')
    parser.add_argument('--output_dir', metavar='DIR', default=DEFAULT_DIR,
                        help='directory where final images are to be saved')
    parser.add_argument('--bypass_divadid', action='store_true',
                        help="do not pass images through DivaDID")

    args = parser.parse_args()

    print("Generating {} images in {}".format(args.output_count,
                                              args.output_dir))

    # generate_single_image({'iter': 0, 'args': args})
    pool = Pool()

    pool.map(generate_single_image,
             list(map(lambda x: {'iter': x, 'args': args},
                      range(args.output_count))))

    print("Generated {} images in {}".format(args.output_count,
                                              args.output_dir))


if __name__ == "__main__":
    main()
