import argparse
import cv2
import hashlib
import numpy as np
import os
import random
import subprocess
import sys
import shutil

from lxml import etree
from PIL import ImageFont, ImageDraw, Image

class Document:
    def __init__(self, stain_level=1, noise_level=1, seed=None):
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

        self.left_margin = 10
        self.top_margin = 10
        self.word_spacing = 10
        self.line_spacing = 10
        self.stain_strength_low_bound = 0.1 * self.stain_level
        self.stain_strength_high_bound = 0.1 + 0.1 * self.stain_level
        self.stain_density_low_bound = 2 + 0.1 * self.stain_level
        self.stain_density_high_bound = 2 + 0.1 * self.stain_level
        self.word_horizontal_shear_scale = 5 + 2 * self.text_noisy_level
        self.word_vertical_shear_scale = 5 + 2 * self.text_noisy_level
        self.word_rotation_scale = 5 + 2 * self.text_noisy_level
        self.word_color_jitter_sigma = 1 + 0.1 * self.text_noisy_level
        self.word_elastic_sigma = 5 - 0.2 * self.text_noisy_level
        self.word_blur_sigma_low_bound = 0.5 + 0.1 * self.text_noisy_level
        self.word_blur_sigma_high_bound = 1 + 0.1 * self.text_noisy_level
        self.word_margin = 5
        self.result = None

        if seed != None:
            self.random_seed = seed
        else:
            # Make sure seed is set, in case multiple proccesses/threads
            random.seed()
            self.random_seed = random.randint(10000, 100000)

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        print("DEBUG: Using seed {}".format(self.random_seed))

        self.gather_data_sources()


    def gather_data_sources(self):
        self.transformed_words_dest_path = "data/transformed_words/"

        #Read a file to load word images
        self.word_image_location_file = open("paths/word_image_folder_paths.txt","r")
        self.word_image_folder_list = self.word_image_location_file.readlines()

        for idx, item in enumerate(self.word_image_folder_list):
            self.word_image_folder_list[idx] = item.rstrip('\r\n')

        #Read a file to load background images
        self.bg_image_location_file = open("paths/word_bg_folder_paths.txt","r")
        self.bg_image_folder_list = self.bg_image_location_file.readlines()

        for idx, item in enumerate(self.bg_image_folder_list):
            self.bg_image_folder_list[idx] = item.rstrip('\r\n')

        #Read a file to load stain paths
        self.stain_paths_file = open("paths/stain_folder_paths.txt","r")
        self.stain_paths_list = self.stain_paths_file.readlines()

        for idx, item in enumerate(self.stain_paths_list):
            self.stain_paths_list[idx] = item.rstrip('\r\n')


    def create(self):
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

        base_working_dir = "/dev/shm/"

        # Get a random background image
        bg_rand_folder = random.choice(self.bg_image_folder_list)
        bg_image_name = random.choice(os.listdir(bg_rand_folder))

        bg_full_path = bg_rand_folder + bg_image_name

        # Generate XML for DivaDID and then degrade background image
        print("- Generating degraded image - pass 1")
        first_xml, first_out = self.generate_degradation_xml(bg_full_path, 1, True, base_working_dir)
        # subprocess.check_call(["java", "-jar", "DivaDid.jar", first_xml], stdout=subprocess.DEVNULL)

        # Add text to degraded background image
        print("- Adding text to image")
        img = Image.open(first_out)
        text_augmented_img = self.add_text(img)
        text_augmented_img.save("/dev/shm/sadfasdfasdf.png")

        # Generate XML for second pass of DivaDID. Degrade image with text
        print("- Generating degraded image - pass 2")
        second_xml, second_out = self.generate_degradation_xml("/dev/shm/sadfasdfasdf.png", 2, True, base_working_dir)
        # subprocess.check_call(["java", "-jar", "DivaDid.jar", second_xml], stdout=subprocess.DEVNULL)

        # self.result = second_out
        self.result = "/dev/shm/sadfasdfasdf.png"


    def save(self, file):
        """
        Save the generated document to the passed location.

        Note that due to the use of DivaDID, for performance reasons,
        intermediate stages of the document generation process are saved at
        /dev/shm. After everything is finished, the resulting product will
        likely need to be moved from that location to a final folder.
        """

        if self.result is None:
            print("Trying to save document before it has been generated.", file=sys.stderr)
            return

        print("File saved to {}".format(file))
        shutil.copy2(self.result, file)


    def white_to_alpha(self, img, color=[0, 0, 0]):

        img[:,:,3] = 255 - img[:,:,0]

        img[:,:,0:3] = color


    def add_text(self, img):
        """
        Add text samples to given image.

        """

        not_done = True
        height_scale = -1

        background_width = img.size[0]
        background_height = img.size[1]

        ground_truth = Image.new("l", (background_width, background_height), 1)

        num_lines = np.rint(np.clip(np.random.normal(15, 4), 3, 30))
        text_start_position = np.clip(np.random.normal(0, 0.2, 2), 0.01, 1)
        text_end_position = np.clip(np.random.normal(1, 0.2, 2), 0, 1)
        space_between_words = int(np.clip(np.random.normal(1, 0.1, 1), 0, 0.05) * background_width)

        avg_line_vertical_scale = (text_end_position[0] - text_start_position[0]) / num_lines
        avg_line_height = background_height * avg_line_vertical_scale

        img = img.convert("RGBA")

        word_rand_folder = random.choice(self.word_image_folder_list)
        word_image_name = random.choice(os.listdir(word_rand_folder))
        word_full_path = word_rand_folder + word_image_name

        word = Image.open(word_full_path)
        word_height = word.size[1]

        avg_word_scale_factor = avg_line_height / word_height

        x_offset = int(np.rint(text_start_position[0] * background_width))
        y_offset = int(np.rint(text_start_position[1] * background_height))

        print("DEBUG: text_start_position {}".format(text_start_position))
        print("DEBUG: x_offset {} y_offset {}".format(x_offset, y_offset))

        # while(not_done):
        for counter in range(7):
            print(x_offset, y_offset)

            word_image_name = random.choice(os.listdir(word_rand_folder))
            word_full_path = word_rand_folder + word_image_name

            word = Image.open(word_full_path)
            word = word.convert("RGBA")
            new_word_width = int(np.rint(word.size[0] * avg_word_scale_factor))
            new_word_height = int(np.rint(word.size[1] * avg_word_scale_factor))

            if x_offset + new_word_width > int(np.rint(text_end_position[1] * background_height)):
                x_offset = int(np.rint(text_start_position[0] * background_width))
                y_offset += avg_line_height

            word = word.resize((new_word_width, new_word_height), Image.BICUBIC)

            word = np.asarray(word).copy()

            self.white_to_alpha(word)

            word = Image.fromarray(word)

            word = word.crop((-x_offset, -y_offset, background_width - x_offset, background_height - y_offset))

            img = Image.alpha_composite(img, word)

            x_offset += new_word_width + space_between_words

        return img

    def generate_degradation_xml(self, base_image, index=0, save=False, save_location=None):
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
        xml_file_name = "degradation_script_{}_{}.xml".format(self.random_seed, index)

        output_full_path = "data/output/" + output_file_name

        if save_location is None:
            xml_full_path = "data/xml/" + xml_file_name
        else:
            xml_full_path = save_location + xml_file_name

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
        copy_e2.set("ref","my-image")

        # Add stains
        for stain_folder in self.stain_paths_list:
            gradient_degradation_e = etree.SubElement(root, "gradient-degradations")
            gradient_degradation_e.set("ref", "my-copy")
            strength_e = etree.SubElement(gradient_degradation_e, "strength")
            strength_e.text = "{:.2f}".format(random.uniform(self.stain_strength_low_bound, self.stain_strength_high_bound))
            density_e = etree.SubElement(gradient_degradation_e, "density")
            density_e.text = "{:.2f}".format(random.uniform(self.stain_density_low_bound, self.stain_density_high_bound))
            iterations_e = etree.SubElement(gradient_degradation_e, "iterations")
            iterations_e.text = "750"
            source_e = etree.SubElement(gradient_degradation_e, "source")
            source_e.text = stain_folder

        save_e = etree.SubElement(root, "save")
        save_e.set("ref", "my-copy")
        save_e.set("file", output_full_path)

        if save == True:
            output_xml = open(xml_full_path, 'w')
            output_xml.write(etree.tostring(root, pretty_print=True).decode("utf-8"))

            return xml_full_path, output_full_path
        else:
            return root


    def print_parameters(self):
        print("\n")
        print("--Parameters--")
        print("\tleft_margin: {}".format(self.left_margin))
        print("\ttop_margin: {}".format(self.top_margin))
        print("\tword_spacing: {}".format(self.word_spacing))
        print("\tline_spacing: {}".format(self.line_spacing))
        print("\tstain_strength_range: [{}, {}]".format(self.stain_strength_low_bound, self.stain_strength_high_bound))
        print("\tstain_density_range: [{}, {}]".format(self.stain_density_low_bound, self.stain_density_high_bound))
        print("\tword_horizontal_shear_scale: {}".format(self.word_horizontal_shear_scale))
        print("\tword_vertical_shear_scale: {}".format(self.word_vertical_shear_scale))
        print("\tword_rotation_scale: {}".format(self.word_rotation_scale))
        print("\tword_color_jitter_sigma: {}".format(self.word_color_jitter_sigma))
        print("\tword_elastic_sigma: {}".format(self.word_elastic_sigma))
        print("\tword_margin: {}".format(self.word_margin))
        print("\tword_blur_sigma_range: [{}, {}]".format(self.word_blur_sigma_low_bound, self.word_blur_sigma_high_bound))
        print("\n")

