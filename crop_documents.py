#!/usr/bin/env python3
"""
A script used to take a set of generated images and convert to LMDBs

This script works in multiple phases. From a base directory which must include
the image files as well as ground truth files, the following steps are taken:

    1) Arbitrarily-sized images are cropped to 256x256 patches.
       Companion images are generated to provide extra data during training
       (such as recall weights)

    2) Generated images are partitioned into a train-val-test set.

    3) Images in each set are then packed into LMDBs.

    4) (OPTIONAL) Data set gets copied to the needed destination.

    5) (OPTIONAL) A new net directory can be created to allow for a new
       experiment.

This script is currently rather inflexible and makes a ton of assumptions about
file hierarchies.
"""
import argparse
import errno
import io
import os
import random
import re
import shutil
import sys
import traceback

import caffe.proto.caffe_pb2
import cv2
import lmdb
import matplotlib.pyplot as plt
import numpy as np

from natsort import natsorted
from multiprocessing import Pool
from termcolor import colored

if os.geteuid == 0:
    sys.exit("Please do not run as root")

def get_next_results_folder(base):
    files = [f for f in os.listdir(base) if re.match(r'results-[0-9]+', f)]

    files = natsorted(files)

    if len(files) == 0:
        return os.path.join(base, "results-001")

    last_number = int(re.match(r'results-([0-9]+)', files[-1]).group(1))

    return os.path.join(base, "results-{:0>3d}".format(last_number + 1))

GRAYSCALE = True

DATA_SET = ""
DESTINATION_ROOT = ""
CREATE_PROJECT = False
PROJECT_SUB_REV = ""
PROJECT_SUB_REV_2 = ""
PROJECT_ITER = ""

SKELETON_DIR = ""

ORIGINAL_DIR = ""
RESULTS_DIR = get_next_results_folder("/tmp")

FULL_DIR = os.path.join(RESULTS_DIR, "full")

TRAIN_DIR = os.path.join(RESULTS_DIR, "train")
VAL_DIR = os.path.join(RESULTS_DIR, "val")
TEST_DIR = os.path.join(RESULTS_DIR, "test")

LABELS_DIR = os.path.join(RESULTS_DIR, "labels")

LMDB_DIR = os.path.join(RESULTS_DIR, "lmdb")

# These folders get appended to the respective train/val/test directory
ORIGINAL_SUBDIR = "original_images"
GT_SUBDIR = "processed_gt"
RECALL_SUBDIR = "recall_weights"
PRECISION_SUBDIR = "precision_weights"

NUM_SAMPLES_PERIMAGE = 5

random.seed('hello')

def debug_print(string):
    if __debug__:
        print("DEBUG: {}".format(string))


def insert_value(orig, value):
    orig_with_ext = os.path.splitext(orig)

    return orig_with_ext[0] + "_" + str(value) + orig_with_ext[1]


def update_locations(distance, first, second):
    if distance > 256:
        first += 128
        second += 128
    else:
        first += distance
        second += distance

    return first, second


def convert(args):
    try:
        file = args[0]
        grayscale = args[1]

        if "gt" in file:
            return

        file = os.path.join(ORIGINAL_DIR, file)
        gt_file = insert_value(file, "gt")

        original = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)

        base_original = original.copy()
        base_gt = gt.copy()

        if base_original.shape[0] < 256 or base_original.shape[1] < 256:
            print(colored("Image is too small", 'red'))
            return

        print("Cropping and prepping {} {}".format(file, base_original.shape))

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
                base_gt = gt.copy()

                cropped_partition_original = base_original[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                cropped_partition_gt = base_gt[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                cropped_partition_gt = np.clip(cropped_partition_gt, 0, 1)
                cropped_partition_gt = 1 - cropped_partition_gt

                if np.count_nonzero(cropped_partition_gt) > 10:
                    file = os.path.join(FULL_DIR, ORIGINAL_SUBDIR, os.path.basename(file))
                    gt_file = os.path.join(FULL_DIR, GT_SUBDIR, os.path.basename(file))
                    recall_file = os.path.join(FULL_DIR, RECALL_SUBDIR, os.path.basename(file))
                    precision_file = os.path.join(FULL_DIR, PRECISION_SUBDIR, os.path.basename(file))

                    weighted_image = 128 * np.ones_like(cropped_partition_original)

                    iter = "{}_{}".format(y_block, x_block)
                    cv2.imwrite(insert_value(file, iter), cropped_partition_original)
                    cv2.imwrite(insert_value(gt_file, iter), cropped_partition_gt)
                    cv2.imwrite(insert_value(recall_file, iter), weighted_image)
                    cv2.imwrite(insert_value(precision_file, iter), weighted_image)


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

    except Exception:
        traceback.print_exc()
        raise


def verify_file(file):
    try:
        image = cv2.imread(file)

        if image.shape != (256, 256):
            raise AssertionError("Image {} was invalid".format(file))

    except Exception:
        debug_print("Removing {}".format(file))
        shutil.remove(file)


def split_into_sets():

    # Since there is a 1:1 between the original files and each type of processed
    # image, we can just iterate over the original files and move each corresponding
    # processed image at the same time.
    files = os.listdir(os.path.join(FULL_DIR, ORIGINAL_SUBDIR))

    sequence = list(range(len(files)))
    random.shuffle(sequence)

    # Use 60% of data as training set, 20% as validation set, and 20% as test set
    train_cut_off = len(files) // (1 / .6)
    val_cut_off = len(files) // (1 / .8)

    # Go through generated images and seperate them into three folders for
    # training, testing, and validation
    for count, index in enumerate(sequence):

        file = files[index]

        original_source = os.path.join(FULL_DIR, ORIGINAL_SUBDIR, file)
        gt_source = os.path.join(FULL_DIR, GT_SUBDIR, file)
        recall_source = os.path.join(FULL_DIR, RECALL_SUBDIR, file)
        precision_source = os.path.join(FULL_DIR, PRECISION_SUBDIR, file)

        if count < train_cut_off:
            target = TRAIN_DIR
        elif count < val_cut_off:
            target = VAL_DIR
        else:
            target = TEST_DIR

        original_dest = os.path.join(target, ORIGINAL_SUBDIR, file)
        gt_dest = os.path.join(target, GT_SUBDIR, file)
        recall_dest = os.path.join(target, RECALL_SUBDIR, file)
        precision_dest = os.path.join(target, PRECISION_SUBDIR, file)

        shutil.move(original_source, original_dest)
        shutil.move(gt_source, gt_dest)
        shutil.move(recall_source, recall_dest)
        shutil.move(precision_source, precision_dest)


    # Generate Label files
    for dir in [ ("train", TRAIN_DIR), ("test", TEST_DIR), ("val", VAL_DIR) ]:
        with open(os.path.join(LABELS_DIR, dir[0] + ".txt"), 'w') as output:

            for file in os.listdir(os.path.join(dir[1], ORIGINAL_SUBDIR)):

                output.write("./{}\n".format(file))

def process_im(im_file):
    im = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
    return im


def open_db(db_file):
    env = lmdb.open(db_file, readonly=False, map_size=int(2 ** 38), writemap=False, max_readers=10000)
    txn = env.begin(write=True)
    return env, txn

def package(im, encoding='png'):
    doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
    datum_im = doc_datum.image

    datum_im.channels = im.shape[2] if len(im.shape) == 3 else 1
    datum_im.width = im.shape[1]
    datum_im.height = im.shape[0]
    datum_im.encoding = 'png'

    # image data
    if encoding != 'none':
        buf = io.BytesIO()
        if datum_im.channels == 1:
            plt.imsave(buf, im, format=encoding, vmin=0, vmax=255, cmap='gray')
        else:
            plt.imsave(buf, im, format=encoding, vmin=0, vmax=1)
        datum_im.data = buf.getvalue()
    else:
        pix = im.transpose(2, 0, 1)
        datum_im.data = pix.tostring()

    return doc_datum

def create_lmdb(images, db_file):
    env, txn = open_db(db_file)
    for x, imname in enumerate(sorted(os.listdir(images))):
        if x and x % 10 == 0:
            print ("Processed {} images".format(x))
        try:
            im_file = os.path.join(images, imname)
            im = process_im(im_file)

            doc_datum = package(im)

            key = "%d:%d:%s" % (76547000 + x * 37, x, os.path.splitext(os.path.basename(im_file))[0])
            # TODO - is this the right encode direction?
            txn.put(key.encode(), doc_datum.SerializeToString())
            if x % 10 == 0:
                txn.commit()
                env.sync()
                print(env.stat())
                print(env.info())
                txn = env.begin(write=True)

        except Exception as e:
            print(e)
            print(traceback.print_exc(file=sys.stdout))
            print("Error occured on:", im_file)
            raise


    print("Done Processing Images")
    txn.commit()
    env.close()


def set_up_lmdbs(args):
    dir = args[0]
    subdir = args[1]

    # Strip off last slash
    type_name = subdir

    lmdb_folder = "{}_{}_lmdb".format( type_name, dir)
    lmdb_folder = os.path.join(LMDB_DIR, subdir, lmdb_folder)

    try:
        debug_print("Creating folder: {}".format(lmdb_folder))
        os.makedirs(lmdb_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    create_lmdb(os.path.join(RESULTS_DIR, dir, subdir), lmdb_folder)


def move_image_to_dest(args):
    src_file = args[0]
    dest = args[1]

    final_destinaion = os.path.join(DESTINATION_ROOT, "data", DATA_SET, dest)

    shutil.copy2(src_file, final_destinaion)


def copy_files_to_position():

    for source in [ORIGINAL_SUBDIR, RECALL_SUBDIR, PRECISION_SUBDIR]:

        dest_dir = os.path.join(DESTINATION_ROOT, "data", DATA_SET, source)

        if os.path.exists(dest_dir):
            print(colored("Deleting {}".format(dest_dir), 'red'))

            shutil.rmtree(dest_dir)

        try:
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        train_sources = [os.path.join(TRAIN_DIR, source, x) for x in os.listdir(os.path.join(TRAIN_DIR, source))]
        test_sources = [os.path.join(TEST_DIR, source, x) for x in os.listdir(os.path.join(TEST_DIR, source))]
        val_sources = [os.path.join(VAL_DIR, source, x) for x in os.listdir(os.path.join(VAL_DIR, source))]

        train_sources = list(map(lambda x: [x, source], train_sources))
        test_sources = list(map(lambda x: [x, source], test_sources))
        val_sources = list(map(lambda x: [x, source], val_sources))

        pool.map(move_image_to_dest, train_sources)
        pool.map(move_image_to_dest, test_sources)
        pool.map(move_image_to_dest, val_sources)

    for dir in [ "train", "val", "test" ]:
        for subdir in [ ORIGINAL_SUBDIR, GT_SUBDIR, RECALL_SUBDIR, PRECISION_SUBDIR ]:
            folder_name = subdir + "_" + dir + "_lmdb"
            dest_dir =  os.path.join(DESTINATION_ROOT, "compute/lmdb", DATA_SET, "256", subdir, folder_name)

            if os.path.exists(dest_dir):
                print(colored("Deleting {}".format(dest_dir), 'red'))

                shutil.rmtree(dest_dir)

            try:
                os.makedirs(dest_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            shutil.copy2(
                os.path.join(LMDB_DIR, subdir, folder_name, "data.mdb"),
                dest_dir)

            shutil.copy2(
                os.path.join(LMDB_DIR, subdir, folder_name, "lock.mdb"),
                dest_dir)

    dest_dir = os.path.join(DESTINATION_ROOT, "data", DATA_SET, "labels")

    if os.path.exists(dest_dir):
        print(colored("Deleting {}".format(dest_dir), 'red'))

        shutil.rmtree(dest_dir)

    try:
        os.makedirs(dest_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    for dir in [ "train.txt", "val.txt", "test.txt" ]:

        shutil.copy2(os.path.join(LABELS_DIR, dir), dest_dir)


def create_project():
    project_subdir = os.path.join(DATA_SET, PROJECT_SUB_REV, PROJECT_SUB_REV_2, PROJECT_ITER)
    net_dir = os.path.join(DESTINATION_ROOT, "nets", project_subdir)

    for file in os.listdir(SKELETON_DIR):
        src = os.path.join(SKELETON_DIR, file)
        dest = os.path.join(net_dir, file)

        try:
            shutil.copytree(src, dest)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    regex_data_set = re.compile(r'DATA_SET')
    regex_project_dir = re.compile(r'DATA_SET')

    for file in os.listdir(net_dir):
        if os.path.isdir(os.path.join(net_dir, file)):
            continue

        with open(net_dir + file) as next_file:
            file_content = next_file.read()

        if file == "solver.prototxt":
            results = regex_data_set.subn(project_subdir, file_content)
        else:
            results = regex_project_dir.subn(DATA_SET, file_content)

        if results[1] > 0:
            with open(os.path.join(net_dir, file)) as next_file:
                next_file.write(results[0])

    try:
        os.makedirs(net_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


parser = argparse.ArgumentParser(description="Crop prepared data files and pack \
                                 into LMDB files")
parser.add_argument('--experiment', default="", nargs=1)
parser.add_argument('source')
parser.add_argument('data_set')
parsed = parser.parse_args()

if DESTINATION_ROOT == "":
    print("Please set DESTINATION_ROOT")
    exit()

DATA_SET = parsed.data_set
ORIGINAL_DIR = parsed.source


print("Source Dir: {}".format(ORIGINAL_DIR))
print("Results Dir: {}".format(RESULTS_DIR))

print("Cleaning destination folder")

try:
    shutil.rmtree(RESULTS_DIR)
except OSError as e:
    if e.errno != errno.ENOENT:
        raise
    else:
        debug_print("Results Dir does not already exist")


# STEP 0 - Make sure needed directories all exist
for dir in [ FULL_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR ]:
    for subdir in [ ORIGINAL_SUBDIR, GT_SUBDIR, RECALL_SUBDIR, PRECISION_SUBDIR ]:
        try:
            full_path = os.path.join(dir, subdir)
            debug_print("Creating folder: {}".format(full_path))
            os.makedirs(full_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

try:
    debug_print("Creating folder: {}".format(LABELS_DIR))
    os.makedirs(LABELS_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

pool = Pool(1)

# STEP 1 - Resize the original images and generate auxiliary files
print("-- Starting STEP 1 --")

pool.map(convert, list(map(lambda x: [x, True], os.listdir(ORIGINAL_DIR))))

# STEP 2 - Generate the recall and precision weights
print("-- Starting STEP 2 --")

split_into_sets()

# STEP 3 - Generate needed lmdb's
print("-- Starting STEP 3 --")

lmdb_dirs = []
for dir in [ "train", "val", "test" ]:
    for subdir in [ ORIGINAL_SUBDIR, GT_SUBDIR, RECALL_SUBDIR, PRECISION_SUBDIR ]:
        lmdb_dirs.append((dir, subdir))

pool.map(set_up_lmdbs, lmdb_dirs)

# STEP 4 - Copy files to needed locations - Optional

if DATA_SET is not None:
    print("-- Starting STEP 4 --")
    copy_files_to_position()
else:
    print("-- SKIPPING STEP 4 --")

# STEP 5 - Set up project folder - Optional

if CREATE_PROJECT is True and DATA_SET is not None:
    print("-- Starting STEP 5 --")
    create_project()
else:
    print("-- SKIPPING STEP 5 --")

