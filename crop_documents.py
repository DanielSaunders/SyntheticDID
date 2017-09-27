#!/usr/bin/env python3
import argparse
import errno
import io
import os
import random
import shutil
import sys
import traceback

import caffe.proto.caffe_pb2
import cv2
import lmdb
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

if os.geteuid == 0:
    sys.exit("Please do not run as root")

GRAYSCALE = True

DATA_SET = None
DESTINATION_ROOT = None
CREATE_PROJECT = False
PROJECT_SUB_REV = ""
PROJECT_SUB_REV_2 = ""
PROJECT_ITER = ""

SKELETON_DIR = None

ORIGINAL_DIR = None
RESULTS_DIR = None
GARBAGE_DIR = None

FULL_DIR = RESULTS_DIR + "full/"

TRAIN_DIR = RESULTS_DIR + "train/"
VAL_DIR = RESULTS_DIR + "val/"
TEST_DIR = RESULTS_DIR + "test/"

LABELS_DIR = RESULTS_DIR + "labels/"

LMDB_DIR = RESULTS_DIR + "lmdb/"

# These folders get appended to the respective train/val/test directory
ORIGINAL_SUBDIR = "original_images/"
GT_SUBDIR = "processed_gt/"
RECALL_SUBDIR = "recall_weights/"
PRECISION_SUBDIR = "precision_weights/"

NUM_SAMPLES_PERIMAGE = 5

random.seed('hello')

def debug_print(string):
    if __debug__:
        print("DEBUG: {}".format(string))

def insert_value(orig, value):
    return orig[:-4] + "_" + str(value) + orig[-4:]


def convert(args):
    try:
        file = args[0]
        grayscale = args[1]

        if "gt" in file:
            return

        file = ORIGINAL_DIR + file
        gt_file = file[:-4] + "_gt" + file[-4:]

        base_original = cv2.imread(file)
        base_gt = cv2.imread(gt_file)

        if base_original.shape[0] < 256 or base_original.shape[1] < 256:
            return

        print("Croppping and prepping {} {}".format(file, base_original.shape))

        for iter in range(NUM_SAMPLES_PERIMAGE):
            original = base_original.copy()
            gt = base_gt.copy()

            for x in range(5):
                top_left_x = random.randint(0, original.shape[1] - 256)
                top_left_y = random.randint(0, original.shape[0] - 256)
                bottom_right_x = top_left_x + 256
                bottom_right_y = top_left_y + 256

                old_original = original.copy()
                old_gt = gt.copy()

                original = original[top_left_y:bottom_right_y, top_left_y:bottom_right_y]
                gt = gt[top_left_y:bottom_right_y, top_left_y:bottom_right_y]
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                gt = np.clip(gt, 0, 1)

                edges = cv2.Canny(original, 100, 200)

                pixel_count = cv2.countNonZero(edges)

                if pixel_count > 0.01 * original.shape[0] * original.shape[1]:
                    break
                elif x == 4 and GARBAGE_DIR is not None:
                    file = GARBAGE_DIR + os.path.basename(file)
                    gt_file = GARBAGE_DIR + os.path.basename(gt_file)


                    cv2.imwrite(file, original)
                    cv2.imwrite(gt_file, gt)
                    return

                original = old_original
                gt = old_gt

            file = FULL_DIR + ORIGINAL_SUBDIR + os.path.basename(file)
            gt_file = FULL_DIR + GT_SUBDIR + os.path.basename(file)
            recall_file = FULL_DIR + RECALL_SUBDIR + os.path.basename(file)
            precision_file = FULL_DIR + PRECISION_SUBDIR + os.path.basename(file)

            # TODO: Why are some grayscale?
            if grayscale == True and len(original.shape) == 3:
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

            weighted_image = 128 * np.ones_like(original)

            if original is None or gt is None or weighted_image is None:
                return

            if original.shape != (256, 256) or gt.shape != (256, 256) or weighted_image.shape != (256, 256):
                return

            cv2.imwrite(insert_value(file, iter), original)
            cv2.imwrite(insert_value(gt_file, iter), gt)
            cv2.imwrite(insert_value(recall_file, iter), weighted_image)
            cv2.imwrite(insert_value(precision_file, iter), weighted_image)
    except Exception:
        traceback.print_exc()
        raise


def verify_file(file):
    try:
        image = cv2.imread(file)

        if image.shape != (256, 256):
            raise AssertionError("Image {} was invalid".format(file))

    except Exception:
        print("Removing {}".format(file))
        shutil.remove(file)


def split_into_sets():

    # Since there is a 1:1 between the original files and each type of processed
    # image, we can just iterate over the original files and move each corresponding
    # processed image at the same time.
    files = os.listdir(FULL_DIR + ORIGINAL_SUBDIR)

    sequence = list(range(len(files)))
    random.shuffle(sequence)

    # Use 60% of data as training set, 20% as validation set, and 20% as test set
    train_cut_off = len(files) // (1 / .6)
    val_cut_off = len(files) // (1 / .8)

    # Go through generated images and seperate them into three folders for
    # training, testing, and validation
    for count, index in enumerate(sequence):

        file = files[index]

        if count < train_cut_off:
            shutil.move(FULL_DIR + ORIGINAL_SUBDIR + file, TRAIN_DIR + ORIGINAL_SUBDIR + file)
            shutil.move(FULL_DIR + GT_SUBDIR + file, TRAIN_DIR + GT_SUBDIR + file)
            shutil.move(FULL_DIR + RECALL_SUBDIR + file, TRAIN_DIR + RECALL_SUBDIR + file)
            shutil.move(FULL_DIR + PRECISION_SUBDIR + file, TRAIN_DIR + PRECISION_SUBDIR + file)
        elif count < val_cut_off:
            shutil.move(FULL_DIR + ORIGINAL_SUBDIR + file, VAL_DIR + ORIGINAL_SUBDIR + file)
            shutil.move(FULL_DIR + GT_SUBDIR + file, VAL_DIR + GT_SUBDIR + file)
            shutil.move(FULL_DIR + RECALL_SUBDIR + file, VAL_DIR + RECALL_SUBDIR + file)
            shutil.move(FULL_DIR + PRECISION_SUBDIR + file, VAL_DIR + PRECISION_SUBDIR + file)
        else:
            shutil.move(FULL_DIR + ORIGINAL_SUBDIR + file, TEST_DIR + ORIGINAL_SUBDIR + file)
            shutil.move(FULL_DIR + GT_SUBDIR + file, TEST_DIR + GT_SUBDIR + file)
            shutil.move(FULL_DIR + RECALL_SUBDIR + file, TEST_DIR + RECALL_SUBDIR + file)
            shutil.move(FULL_DIR + PRECISION_SUBDIR + file, TEST_DIR + PRECISION_SUBDIR + file)



    # Generate Label files
    for dir in [ ("train", TRAIN_DIR), ("test", TEST_DIR), ("val", VAL_DIR) ]:
        with open("{}{}.txt".format(LABELS_DIR, dir[0]), 'w') as output:

            for file in os.listdir(dir[1] + ORIGINAL_SUBDIR):

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
    for x, imname in enumerate(os.listdir(images)):
        if x and x % 10 == 0:
            print ("Processed {} images".format(x))
        try:
            im_file = os.path.join(images, imname)
            im = process_im(im_file)
            # remove patches containing all background
            # if remove_background:
                # idx = 0
                # while idx < len(ims):
                    # gt = gts[idx]
                    # if gt.max() == 0:
                        # #print "Deleting patch %d of image %s" % (idx, im_file)
                        # del gts[idx]
                        # del ims[idx]
                    # else:
                        # idx += 1

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
    type_name = subdir[:-1]

    lmdb_folder = "{}{}{}_{}_lmdb".format(LMDB_DIR, subdir, type_name, dir)

    try:
        debug_print("Creating folder: {}".format(lmdb_folder))
        os.makedirs(lmdb_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    create_lmdb(RESULTS_DIR + dir + "/" + subdir, lmdb_folder)


def move_image_to_dest(src_file):
    shutil.copy2(src_file, DESTINATION_ROOT + "/data/" + DATA_SET + "/original_images/")


def copy_files_to_position():
    dest_dir = DESTINATION_ROOT + "/data/" + DATA_SET + "/original_images/"

    try:
        os.makedirs(dest_dir)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise

    pool.map(move_image_to_dest, [TRAIN_DIR + ORIGINAL_SUBDIR + x for x in os.listdir(TRAIN_DIR + ORIGINAL_SUBDIR)])
    pool.map(move_image_to_dest, [TEST_DIR + ORIGINAL_SUBDIR + x for x in os.listdir(TEST_DIR + ORIGINAL_SUBDIR)])
    pool.map(move_image_to_dest, [VAL_DIR + ORIGINAL_SUBDIR + x for x in os.listdir(VAL_DIR + ORIGINAL_SUBDIR)])

    for dir in [ "train", "val", "test" ]:
        for subdir in [ ORIGINAL_SUBDIR, GT_SUBDIR, RECALL_SUBDIR, PRECISION_SUBDIR ]:
            dest_dir =  DESTINATION_ROOT + "/compute/lmdb/" + DATA_SET + "256/" + subdir + subdir[:-1] + "_" + dir + "_lmdb/"

            try:
                os.makedirs(dest_dir)
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise

            shutil.copy2(
                LMDB_DIR + subdir + subdir[:-1] + "_" + dir + "_lmdb/" + "data.mdb",
                dest_dir)

            shutil.copy2(
                LMDB_DIR + subdir + subdir[:-1] + "_" + dir + "_lmdb/" + "lock.mdb",
                dest_dir)

    for dir in [ "train.txt", "val.txt", "test.txt" ]:
        dest_dir =  DESTINATION_ROOT + "/data/" + DATA_SET + "/labels/"

        try:
            os.makedirs(dest_dir)
        except OSError, e:
            if e.errno != errno.EEXIST:
                raise

        shutil.copy2(LABELS_DIR + dir, dest_dir)


def create_project():
    project_subdir = "{}/{}/{}/{}/".format(
        DATA_SET
        PROJECT_SUB_REV,
        PROJECT_SUB_REV_2,
        PROJECT_ITER)

    project_subdir = re.sub(r'\/+', '/', project_subdir)

    net_dir = "{}/nets/{}/".format(
        DESTINATION_ROOT,
        project_subdir)

    for file in os.listdir(SKELETON_DIR):
        src = SKELETON_DIR + file
        dest = net_dir + file

        try:
            shutil.copytree(src, dest)
        except OSError, e:
            if e.errno != errno.EEXIST:
                raise

    regex_data_set = re.compile(r'DATA_SET')
    regex_project_dir = re.compile(r'DATA_SET')

    for file in os.listdir(net_dir):
        if os.path.isdir(net_dir + file):
            continue

        with open(net_dir + file) as next_file:
            file_content = next_file.read()

        if file == "solver.prototxt":
            results = regex_data_set.subn(project_subdir, file_content)
        else:
            results = regex_project_dir.subn(DATA_SET, file_content)

        if results[1] > 0:
            with open(net_dir + file) as next_file:
                next_file.write(results[0])

    try:
        os.makedirs(net_dir)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise




# parser = argparse.ArgumentParser(description="Prepare generated images for training")
# parser.add_argument('data_dir', help="the directory in which the base images reside")
# parser.add_argument('--overwrite-crops', help="crop images, even if cropped images already exist")
# parser.add_argument('--overwrite-val-and-train', help="seperate into val and train, even if directories already exist")
# parser.add_argument('--overwrite-all', help="overwrite everything, no matter what work has already been done")
# parser.add_argument('--grayscale', help="render final images in grayscale")
# parser.add_argument('--color', help="render final images in color")
# parser.parse_args()

print("Source Dir: {}".format(ORIGINAL_DIR))
print("Results Dir: {}".format(RESULTS_DIR))

print("Cleaning destination folder")
shutil.rmtree(RESULTS_DIR)

# STEP 0 - Make sure needed directories all exist
for dir in [ FULL_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR ]:
    for subdir in [ ORIGINAL_SUBDIR, GT_SUBDIR, RECALL_SUBDIR, PRECISION_SUBDIR ]:
        try:
            debug_print("Creating folder: {}".format(dir + subdir))
            os.makedirs(dir + subdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

try:
    debug_print("Creating folder: {}".format(LABELS_DIR))
    os.makedirs(LABELS_DIR)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

pool = Pool()

# STEP 1 - Resize the original images and generate auxiliary files
print("-- Starting STEP 1 --")

pool.map(convert, list(map(lambda x: [x, True], os.listdir(ORIGINAL_DIR))))

print("-- Starting STEP 1b --")

all_directories = []
all_directories += os.listdir(FULL_DIR + ORIGINAL_SUBDIR)
all_directories += os.listdir(FULL_DIR + GT_SUBDIR)
all_directories += os.listdir(FULL_DIR + RECALL_SUBDIR)
all_directories += os.listdir(FULL_DIR + PRECISION_SUBDIR)

pool.map(verify_file, all_directories)

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
print("-- Starting STEP 4 --")

if DATA_SET is not None:
    copy_files_to_position()

# STEP 5 - Set up project folder - Optional
print("-- Starting STEP 5 --")

if CREATE_PROJECT is True and DATA_SET is not None:
    create_project()

