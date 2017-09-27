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

SOURCE="/home/saund9/git/CycleGAN/data_d/first_stage/"
DEST="/home/saund9/git/CycleGAN/data_d/trainB/"
NUM_SAMPLES_PERIMAGE=10

def insert_value(orig, value):
    return orig[:-4] + "_" + str(value) + orig[-4:]

def verify_file(file):
    try:
        image = cv2.imread(file)

        if image.shape != (256, 256):
            raise AssertionError("Image {} was invalid".format(file))

    except Exception:
        print("Removing {}".format(file))
        shutil.remove(file)

def convert(file):
    file = SOURCE + file
    print(file)
    base_original = cv2.imread(file)

    if base_original.shape[0] < 256 or base_original.shape[1] < 256:
        return

    print("Croppping {} {}".format(file, base_original.shape))

    for iter in range(NUM_SAMPLES_PERIMAGE):
        original = base_original.copy()

        top_left_y = random.randint(0, original.shape[0] - 256)
        top_left_x = random.randint(0, original.shape[1] - 256)
        bottom_right_x = top_left_x + 256
        bottom_right_y = top_left_y + 256

        original = original[top_left_y:bottom_right_y, top_left_y:bottom_right_y]

        file = DEST + os.path.basename(file)
        file = insert_value(file, iter)

        cv2.imwrite(file, original)


pool = Pool()

pool.map(convert, os.listdir(SOURCE))

pool.map(verify, os.listdir(DEST))
