#!/usr/bin/env python3
import argparse
import sys

if sys.version_info < (3, 0):
    sys.stdout.write("Python 2 is not supported. Please use Python 3\n")
    sys.exit(1)

from document import Document

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

parser = argparse.ArgumentParser(description='Generate some images.')
parser.add_argument('output_count', metavar='N', type=check_output_count, nargs='?', default=5,
                    help='number of images to generate')
parser.add_argument('stain_level', metavar='S', type=check_level, nargs='?', default=1,
                    help='amount of noise in stains')
parser.add_argument('text_noise_level', metavar='T', type=check_level, nargs='?', default=1,
                    help='amount of noise in text')

args = parser.parse_args()

print("Generating {} images".format(args.output_count))

for image_count in range(args.output_count):

    print("Generating image #{}".format(image_count + 1))

    document = Document(args.stain_level, args.text_noise_level, seed=1234)

    document.create()
    document.save("text.png")
    exit()

