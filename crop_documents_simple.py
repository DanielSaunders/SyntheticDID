import argparse
import errno
import io
import os
import random
import shutil
import sys

import cv2
import numpy

from multiprocessing import Pool
from termcolor import colored

SOURCE="/data/input_for_trainB/"
DEST="/data/trainB/"
NUM_SAMPLES_PERIMAGE=10

def insert_value(orig, value):
    return orig[:-4] + "_" + str(value) + ".png"

def verify_file(file):
    file = DEST + file
    try:
        image = cv2.imread(file)

        if image.shape[0] != 256 and image.shape[1] != 256:
            raise AssertionError("Image {} was invalid. Shape: {}".format(file, image.shape))

    except Exception as e:
        print(colored(e, 'red'))
        print("Removing {}".format(file))
        os.remove(file)

def update_locations(distance, first, second):
    if distance > 256:
        first += 128
        second += 128
    else:
        first += distance
        second += distance

    return first, second

def convert(file):
    file = SOURCE + file
    print(file)
    original = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    base_original = original.copy()

    if base_original.shape[0] < 256 or base_original.shape[1] < 256:
        print(colored("Image is too small", 'red'))
        return

    top_left_x = 0
    top_left_y = 0
    bottom_right_x = 256
    bottom_right_y = 256
    distance_from_edge = base_original.shape[1] - bottom_right_x
    distance_from_bottom = base_original.shape[0] - bottom_right_y

    x_block = 0
    y_block = 0

    while distance_from_bottom != 0:
        while distance_from_edge != 0:
            base_original = original.copy()
            cropped_partition = base_original[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            new_file = DEST + os.path.basename(file)
            new_file = insert_value(new_file, "{}_{}".format(y_block, x_block))

            cv2.imwrite(new_file, cropped_partition)
            # print("tl: ({}, {}) br: ({}, {}) distance: {}".format(top_left_x, top_left_y, bottom_right_x, bottom_right_y, distance_from_edge))

            x_block += 1

            distance_from_edge = base_original.shape[1] - bottom_right_x

            top_left_x, bottom_right_x = update_locations(distance_from_edge,
                                                          top_left_x,
                                                          bottom_right_x)


        y_block += 1
        x_block = 0
        top_left_x = 0
        bottom_right_x = 256

        distance_from_bottom = base_original.shape[0] - bottom_right_y
        distance_from_edge = base_original.shape[1] - bottom_right_x

        top_left_y, bottom_right_y = update_locations(distance_from_bottom,
                                                      top_left_y,
                                                      bottom_right_y)



pool = Pool()

pool.map(convert, os.listdir(SOURCE))

# pool.map(verify_file, os.listdir(DEST))
