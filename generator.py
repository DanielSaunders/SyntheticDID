import argparse
import cv2
import hashlib
import numpy as np
import os
import random
import sys

from lxml import etree

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

        if seed != None:
            self.random_seed = seed
        else:
            # Make sure seed is set, in case multiple proccesses/threads
            random.seed()
            self.random_seed = random.randint(10000, 100000)

        random.seed(self.random_seed)

        self.print_parameters()

        self.gather_data_sources()

    def gather_data_sources(self):
        #Macro
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

    def generate_base_xml(self):

        self.x_value = self.left_margin
        self.y_value = self.top_margin
        self.root = etree.Element("root")

        #Fill the background with words
        alias_e = etree.SubElement(self.root, "alias")
        alias_e.set("id", "INPUT")

        #Example: <alias id="INPUT" value="test_backgrounds/bg1_resized.png"/>
        bg_rand_folder = random.choice(self.bg_image_folder_list)
        bg_image_name = random.choice(os.listdir(bg_rand_folder))
        alias_e.set("value", bg_rand_folder + bg_image_name)
        bg_img = cv2.imread(bg_rand_folder + bg_image_name)
        bg_image_height, bg_image_width, bg_channels = bg_img.shape

        #Create blank version of the bg
        blank_image = np.zeros((bg_image_height,bg_image_width,3), np.uint8)
        blank_image[::]=(255,255,255)
        cv2.imwrite("data/blank_bgs/"+bg_image_name, blank_image)
        #Example: 
            #<image id="my-image">
            #   <load file="INPUT"/>
            #</image>
        image_e = etree.SubElement(self.root, "image")
        image_e.set("id", "my-image")
        load_e = etree.SubElement(image_e, "load")
        load_e.set("file", "INPUT")
        #Example:
            #<image id="my-copy">
            #   <copy ref="my-image"/>
            #</image>
        image_e2 = etree.SubElement(self.root, "image")
        image_e2.set("id", "my-copy")
        copy_e2 = etree.SubElement(image_e2, "copy")
        copy_e2.set("ref","my-image")
        #Example:
            # #<manual-gradient-degradations ref="my-copy">
        # manual_gradient_degradation_e = etree.SubElement(root, "manual-gradient-degradations")
        # manual_gradient_degradation_e.set("ref", "my-copy")
        # has_v_space = True
        # has_h_space = True
        # max_row_height = 0
        # word_count = 0
        # while(has_v_space == True):
            # #Example:
                # #<degradation>
                # #<file>test_backgrounds/sample_text.png</file>
                # #<strength>1</strength>
                # #<x>0</x>
                # #<y>0</y>
                # #</degradation>
            # word_rand_folder = random.choice(word_image_folder_list)
            # word_image_name = random.choice(os.listdir(word_rand_folder))
            # #Perform random transformations, save in data/transfromed_words
            # generated_image_name = str(output_index) + "_" + str(word_count) + "_" + word_image_name
            # from word_transform import get_random_img_transform
            # #Randomize transform parameters
            # rand_h_shear_dg = random.random()*word_horizontal_shear_scale
            # rand_v_shear_dg = random.random()*word_vertical_shear_scale
            # rand_rotation_dg = random.random()*word_rotation_scale
            # rand_color_jitter_sigma = random.random()*word_color_jitter_sigma
            # rand_elastic_sigma = word_elastic_sigma + (random.random()-1)*0.1
            # rand_blur_sigma = random.uniform(word_blur_sigma_low_bound, word_blur_sigma_high_bound)
            # print("write path:"+transformed_words_dest_path+generated_image_name)
            # rand_word_im = get_random_img_transform(word_rand_folder+word_image_name, \
            # rand_h_shear_dg, rand_v_shear_dg, rand_rotation_dg, rand_color_jitter_sigma, \
            # rand_elastic_sigma, rand_blur_sigma, word_margin)
            # cv2.imwrite(transformed_words_dest_path+generated_image_name, rand_word_im)
            # has_h_space, has_v_space, x_next_value, y_next_value, img_height = is_valid_location(x_value, y_value, word_spacing, line_spacing, max_row_height,\
            # left_margin, top_margin, "data/transformed_words/" + generated_image_name, bg_image_width, bg_image_height)
            # if(img_height > max_row_height):
                # max_row_height = img_height
            # if has_v_space == False:
                # x_value = left_margin
                # y_value = top_margin
                # max_row_height = 0
                # break
            # if has_h_space == False:
                # x_value = x_next_value
                # y_value = y_next_value
                # max_row_height = 0
                # continue

            # degradation_e = etree.SubElement(manual_gradient_degradation_e, "degradation")
            # file_e = etree.SubElement(degradation_e, "file")
            # file_e.text = transformed_words_dest_path+generated_image_name#word_rand_folder + word_image_name
            # strength_e = etree.SubElement(degradation_e, "strength")
            # strength_e.text = "1"

            # x_e = etree.SubElement(degradation_e, "x")
            # x_e.text = str(x_value)
            # y_e = etree.SubElement(degradation_e, "y")
            # y_e.text = str(y_value)
            # x_value = x_next_value
            # y_value = y_next_value

            # word_count += 1
        # #Example:
            # #...
            # #<multi-core/>
            # #<iterations>500</iterations>
            # #</manual-gradient-degradations>
        # multi_core_e = etree.SubElement(manual_gradient_degradation_e, "multi-core")
        # iterations_e = etree.SubElement(manual_gradient_degradation_e, "iterations")
        # iterations_e.text = "500"
        #Add stains
        for stain_folder in self.stain_paths_list:
            #Example:
                #<gradient-degradations ref="my-copy">
                #<strength>1.2</strength>
                #<density>25</density>
                #<iterations>750</iterations>
                #<source>data/spots</source>
                #</gradient-degradations>   
            gradient_degradation_e = etree.SubElement(self.root, "gradient-degradations")
            gradient_degradation_e.set("ref", "my-copy")
            strength_e = etree.SubElement(gradient_degradation_e, "strength")
            strength_e.text = "{:.2f}".format(random.uniform(self.stain_strength_low_bound, self.stain_strength_high_bound))
            density_e = etree.SubElement(gradient_degradation_e, "density")
            density_e.text = "{:.2f}".format(random.uniform(self.stain_density_low_bound, self.stain_density_high_bound))
            iterations_e = etree.SubElement(gradient_degradation_e, "iterations")
            iterations_e.text = "750"
            source_e = etree.SubElement(gradient_degradation_e, "source")
            source_e.text = stain_folder
        #Example:
            #<save ref="my-copy" file="outputs    ext_insertion_test1.png"/>
        save_e = etree.SubElement(self.root, "save")
        save_e.set("ref", "my-copy")
        save_e.set("file", "data/outputs/degraded_"+ str(self.random_seed) + "_"+bg_image_name)

        output_xml = open("data_generator_script.xml", 'w')
        output_xml.write(etree.tostring(self.root, pretty_print=True).decode("utf-8"))

    def generate_bare_text_xml(self):
        for element in self.root.xpath('//gradient-degradations' ) :
            element.getparent().remove(element)

        for element in self.root.findall("alias"):
            old_value = element.get("value")
            splited_value = old_value.split("/")
            splited_value[len(splited_value)-3] = "blank_bgs"
            del splited_value[len(splited_value)-2]
            element.set("value", "/".join(splited_value))

        for element in self.root.findall("save"):
            old_value = element.get("file")
            splitted_value = old_value.split("/")
            splitted_value[len(splitted_value)-1] = splitted_value[len(splitted_value)-1]
            splitted_value[len(splitted_value)-2] = "ground_truths"
            original_img_name = splitted_value[len(splitted_value)-1]
            element.set("file", "/".join(splitted_value))

        white_bg_xml = open("match_generator_script.xml", 'w')
        white_bg_xml.write(etree.tostring(self.root, pretty_print=True).decode("utf-8"))

    def print_parameters(self):
        print("\n")
        print("--Parameters--")
        print("\tleft_margin:", self.left_margin)
        print("\ttop_margin:", self.top_margin)
        print("\tword_spacing:", self.word_spacing)
        print("\tline_spacing:", self.line_spacing)
        print("\tstain_strength_range: [", self.stain_strength_low_bound,", ", self.stain_strength_high_bound,"]")
        print("\tstain_density_range: [", self.stain_density_low_bound,", ", self.stain_density_high_bound,"]")
        print("\tword_horizontal_shear_scale: ", self.word_horizontal_shear_scale)
        print("\tword_vertical_shear_scale: ", self.word_vertical_shear_scale)
        print("\tword_rotation_scale: ", self.word_rotation_scale)
        print("\tword_color_jitter_sigma: ", self.word_color_jitter_sigma)
        print("\tword_elastic_sigma: ", self.word_elastic_sigma)
        print("\tword_margin: ", self.word_margin)
        print("\tword_blur_sigma_range: [", self.word_blur_sigma_low_bound,", ", self.word_blur_sigma_high_bound,"]")
        print("\n")

