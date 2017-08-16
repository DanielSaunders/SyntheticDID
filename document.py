"""
A synthetic handwritten Document

This module includes the Document class.
"""
import errno
import os
import random
import subprocess
import sys
import shutil

import cv2
import image_util as util
import numpy as np

from lxml import etree

HANDWRITTEN_WORDS_DIR = "/data/synthetic/handwriting/iamdb/"
BACKGROUND_IMAGES_DIR = "/data/synthetic/backgrounds/"
STAIN_IMAGES_DIR = "/data/synthetic/spots/"
DEFAULT_OUTPUT_DIR = "/data/"

# /dev/shm should be mounted in RAM - allowing for fast IPC (Used as a
# consequence of using DivaDID.)
TMP_DIR = "/dev/shm/"


class Document:
    """
    A synthetic handwritten Document

    A Document instance is a synthetic, handwritten, text image. This class
    handles the generation of such images. It also has helper functions that
    allow for the saving of the generated images to disk.
    """

    def __init__(self, stain_level=1, noise_level=1,
                 seed=None, output_loc=DEFAULT_OUTPUT_DIR):
        """
        Initialize a new Document

        For every synthetic document created, a new Document object should
        be instantiated.

        A Document object is not guaranteed to be thread- or process-safe.
        However, the Document class itself is safe and different objects can
        be instantiated in different threads or processes. All member functions
        can be safely called without concern about locks.
        """
        self.stain_level = stain_level
        self.text_noisy_level = noise_level

        self.result = None
        self.result_ground_truth = None

        self.output_dir = output_loc
        print(self.output_dir)

        if seed is not None:
            self.random_seed = seed
        else:
            self.assign_random_seed()

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        print("DEBUG: Using seed {}".format(self.random_seed))

        self.gather_data_sources()

    def assign_random_seed(self):
        tries = 0
        condition = True

        random.seed()

        # This roughly emulates a do-while loop. We need to assign a seed at
        # least once.
        while condition:
            if tries > 10:
                raise RuntimeError("Could not find an unused seed")

            self.random_seed = random.randint(10000, 100000)

            file = "img_{}.png".format(self.random_seed)
            file = self.output_dir + '/' + file

            condition = os.path.isfile(file)

            tries += 1

    def gather_data_sources(self):
        """ Parse lists of needed directories. """

        self.word_image_folder_list = []

        for hw_dir in os.listdir(HANDWRITTEN_WORDS_DIR):
            files = os.listdir(HANDWRITTEN_WORDS_DIR + hw_dir)

            new_path = HANDWRITTEN_WORDS_DIR + hw_dir + "/"

            for idx, item in enumerate(files):
                # subfolders = os.listdir(new_path + item)

                # for idx, sub_folder_item in enumerate(subfolders):
                files[idx] = new_path + item + "/"

            self.word_image_folder_list += files

    def create(self, bypass=False):
        """
        Generate a synthetic text document.

        The current generation process has three stages. The first is to pick
        a random background image and then use DivaDID to apply some simply
        degradations to add some noise and natural variation.

        The second stage is to add text to the background image. During this
        process, the "ground truth" file is also created.

        The third and final stage is a second iteration of DivaDID. Now that we
        have text on the document, we degrade the image once more to give it
        a somewhat more realistic appearance.
        """

        base_working_dir = TMP_DIR

        # Get a random background image
        bg_image_name = random.choice(os.listdir(BACKGROUND_IMAGES_DIR))

        bg_full_path = BACKGROUND_IMAGES_DIR + bg_image_name

        if bypass is True:
            print("-{} Adding text to image {} -".format(self.random_seed, bg_full_path))
            img = cv2.imread(bg_full_path)
            text_augmented_img = self.add_text(img)
            cv2.imwrite(base_working_dir + str(self.random_seed) + "_augmented.png",
                        text_augmented_img)

            self.result = base_working_dir + str(self.random_seed) + "_augmented.png"
            return

        # Generate XML for DivaDID and then degrade background image
        print("- Generating degraded image - pass 1")
        first_xml, first_image = self.generate_degradation_xml(bg_full_path,
                                                             1,
                                                             True,
                                                             base_working_dir)

        subprocess.check_call(["java", "-jar", "DivaDid.jar", first_xml],
                              stdout=subprocess.DEVNULL)

        # Add text to degraded background image
        print("-{} Adding text to image {} -".format(self.random_seed, bg_full_path))
        img = cv2.imread(first_image)
        text_augmented_img = self.add_text(img)
        cv2.imwrite(base_working_dir + str(self.random_seed) + "_augmented.png",
                    text_augmented_img)


        # Generate XML for second pass of DivaDID. Degrade image with text
        print("- Generating degraded image - pass 2")
        second_xml, second_image = self.generate_degradation_xml(
            base_working_dir + str(self.random_seed) + "_augmented.png",
            2,
            True,
            base_working_dir)

        subprocess.check_call(["java", "-jar", "DivaDid.jar", second_xml],
                              stdout=subprocess.DEVNULL)

        self.result = second_image

        os.remove(first_xml)
        os.remove(second_xml)
        os.remove(first_image)

    def save(self, file=None):
        """
        Save the generated document to the passed location.

        Note that due to the use of DivaDID, for performance reasons,
        intermediate stages of the document generation process are saved at
        /dev/shm. After everything is finished, the resulting product will
        likely need to be moved from that location to a final folder.
        """

        if self.result is None:
            print("Trying to save document before it has been generated.",
                  file=sys.stderr)
            return

        if file is None:
            file = "img_{}.png".format(self.random_seed)

        file = self.output_dir + '/' + file

        try:
            os.makedirs(self.output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        print("File saved to {}".format(file))
        shutil.copy2(self.result, file)

        os.remove(self.result)

    def save_ground_truth(self, file=None):
        """
        Save the generated document to the passed location.

        Note that due to the use of DivaDID, for performance reasons,
        intermediate stages of the document generation process are saved at
        /dev/shm. After everything is finished, the resulting product will
        likely need to be moved from that location to a final folder.
        """

        if self.result is None:
            print("Trying to save document before it has been generated.",
                  file=sys.stderr)
            return

        if file is None:
            file = "img_{}_gt.png".format(self.random_seed)

        file = self.output_dir + '/' + file

        try:
            os.makedirs(self.output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        print("File saved to {}".format(file))
        shutil.copy2(self.result_ground_truth, file)

    def add_text(self, img):
        """ Add text samples to given image.  """

        not_done = True
        color = np.array((53, 52, 46))

        # TODO
        background_height = img.shape[0]
        background_width = img.shape[1]

        ground_truth = np.ones((background_height, background_width, 3), np.uint8)

        num_lines = np.rint(np.clip(np.random.normal(15, 4), 3, 30))
        text_start_position = np.clip(np.random.normal(0, 0.2, 2), 0.01, 1)
        text_end_position = np.clip(np.random.normal(1, 0.2, 2), 0, 1)
        space_between_words = int(np.clip(np.random.normal(1, 0.1, 1), 0, 0.05) * background_width)

        line_height_variation = np.clip(np.random.normal(1, 0.1, 1), -.5, .5)

        avg_line_vertical_scale = (text_end_position[0] - text_start_position[0]) / num_lines
        avg_line_height = background_height * avg_line_vertical_scale

        # img = util.add_alpha_channel(img)

        word_rand_folder = random.choice(self.word_image_folder_list)
        word_image_name = random.choice(os.listdir(word_rand_folder))
        word_full_path = word_rand_folder + word_image_name

        word = cv2.imread(word_full_path)
        word_height = word.shape[0]
        word_width = word.shape[1]

        avg_word_scale_factor = min(avg_line_height / word_height, 1.0)

        x_offset = int(np.rint(text_start_position[0] * background_width))
        y_offset = int(np.rint(text_start_position[1] * background_height))

        # Add individual words until we run out of space
        while not_done:

            word_image_name = random.choice(os.listdir(word_rand_folder))
            word_full_path = word_rand_folder + word_image_name

            word = cv2.imread(word_full_path)
            word = util.add_alpha_channel(word)

            word_height = word.shape[0]
            word_width = word.shape[1]

            if word.shape[0] == 0 or word.shape[1] == 1:
                continue

            new_word_width = int(np.rint(word.shape[1] * avg_word_scale_factor))
            new_word_height = int(np.rint(word.shape[0] * avg_word_scale_factor))

            if x_offset + new_word_width > int(np.rint(text_end_position[1] * background_width)):
                if y_offset + word_height + new_word_height > int(np.rint(text_end_position[0] * background_height)):
                    break

                x_offset = int(np.rint(text_start_position[0] * background_width))
                # y_offset += int(avg_line_height + (avg_line_height * line_height_variation))
                y_offset += new_word_height

            word = cv2.resize(word, (new_word_width, new_word_height), cv2.INTER_CUBIC)

            color += np.random.randint(-2, 3, size=3)
            util.white_to_alpha(word, color=color)

            word = cv2.copyMakeBorder(word,
                                      y_offset,
                                      background_height - y_offset - new_word_height,
                                      x_offset,
                                      background_width - x_offset - new_word_width,
                                      cv2.BORDER_CONSTANT,
                                      (0, 0, 0, 0))

            ground_truth_word = word.copy()

            img = util.alpha_composite(img, word)
            ground_truth = util.alpha_composite(ground_truth, ground_truth_word)
            # ground_truth.paste(ground_truth_word, mask=ground_truth_word)  # = Image.alpha_composite(ground_truth, ground_truth_word)

            x_offset += new_word_width + space_between_words

        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
        retval, ground_truth = cv2.threshold(ground_truth, 20, 255, cv2.THRESH_BINARY_INV)

        self.result_ground_truth = TMP_DIR + str(self.random_seed) + "_gt.png"

        cv2.imwrite(self.result_ground_truth, ground_truth)

        return img

    def generate_degradation_xml(self,
                                 base_image,
                                 index=0,
                                 save=False,
                                 save_location=None):
        """
        Generate the XML needed by DivaDID to add surface stains to an image.

        This function takes the given base image and creates the XML that will
        be fed to DivaDID which specifies how to add a variety of surface
        stains and other imperfections.

        The generated XML can either be saved to the file system for analysis
        or further usage, or simply returned to be fed directly to DivaDID.

        In either case, the return value is the generated XML.
        """

        output_file_name = "degraded_{}_{}.png".format(self.random_seed, index)
        xml_file_name = "degradation_script_{}_{}.xml".format(self.random_seed,
                                                              index)

        stain_strength_low_bound = 0.1 * self.stain_level
        stain_strength_high_bound = 0.1 + 0.1 * self.stain_level
        stain_density_low_bound = 2 + 0.1 * self.stain_level
        stain_density_high_bound = 2 + 0.1 * self.stain_level

        if save_location is None:
            xml_full_path = "data/xml/" + xml_file_name
            output_full_path = "data/output/" + output_file_name
        else:
            xml_full_path = save_location + xml_file_name
            output_full_path = save_location + output_file_name

        root = etree.Element("root")

        alias_e = etree.SubElement(root, "alias")
        alias_e.set("id", "INPUT")

        alias_e.set("value", base_image)

        image_e = etree.SubElement(root, "image")
        image_e.set("id", "my-image")
        load_e = etree.SubElement(image_e, "load")
        load_e.set("file", "INPUT")

        image_e2 = etree.SubElement(root, "image")
        image_e2.set("id", "my-copy")
        copy_e2 = etree.SubElement(image_e2, "copy")
        copy_e2.set("ref", "my-image")

        # Add stains
        for stain_folder in [STAIN_IMAGES_DIR]:  # os.listdir(STAIN_IMAGES_DIR)[0:20]:
            gradient_degradation_e = etree.SubElement(root,
                                                      "gradient-degradations")
            gradient_degradation_e.set("ref", "my-copy")
            strength_e = etree.SubElement(gradient_degradation_e, "strength")
            strength_e.text = "{:.2f}".format(
                random.uniform(stain_strength_low_bound,
                               stain_strength_high_bound))
            density_e = etree.SubElement(gradient_degradation_e, "density")
            density_e.text = "{:.2f}".format(
                random.uniform(stain_density_low_bound,
                               stain_density_high_bound))
            iterations_e = etree.SubElement(gradient_degradation_e,
                                            "iterations")
            iterations_e.text = "750"
            source_e = etree.SubElement(gradient_degradation_e, "source")
            source_e.text = stain_folder

        save_e = etree.SubElement(root, "save")
        save_e.set("ref", "my-copy")
        save_e.set("file", output_full_path)

        if save is True:
            output_xml = open(xml_full_path, 'w')
            output_xml.write(
                etree.tostring(root, pretty_print=True).decode("utf-8"))

            return xml_full_path, output_full_path

        return root
