import cv2
import numpy as np

class TextWriterState:
    def __init__(self, doc_shape):
        self.shape = np.array(doc_shape, dtype=np.int)

        self.start_position = np.array(self.shape * 0.05, dtype=np.int)[0:2]

        self.end_position = np.array(self.shape * 0.95, dtype=np.int)[0:2]

        self.effective_shape = self.end_position - self.start_position

        self.offset = np.copy(self.start_position)

        self.avg_dist_lines = np.array(self.shape[0] * 0.03, dtype=np.int)
        self.avg_dist_words = np.array(self.shape[1] * 0.02, dtype=np.int)

        self.started = False

        self.last_word_shape = np.array([0, 0], dtype=np.int)

        self.last_line_height = 0

        self.padded_image = None


    def get_next_word_pos(self, word_shape):
        word_shape = word_shape[0:2]

        if np.any(word_shape > self.effective_shape):
            return None

        if self.offset[0] + word_shape[0] > self.end_position[0]:
            return None

        if self.started == False:
            self.started = True
            self.last_word_shape = word_shape
            self.last_line_height = word_shape[0]
            return np.copy(self.offset)

        if self.offset[1] + word_shape[1] + self.avg_dist_words + self.last_word_shape[1] > self.end_position[1]:

            self.offset[1] = self.start_position[1]
            # Ok, the problem with this, is that we are moving the pointer
            # to the next line, assuming the current line has the same height
            # as this new word. What if they are not?
            self.offset[0] += self.last_line_height + self.avg_dist_lines

            self.last_line_height = 0

            if self.offset[0] + word_shape[0] > self.end_position[0]:
                return None
        else:
            self.offset[1] += self.last_word_shape[1] + self.avg_dist_words

        if word_shape[0] > self.last_line_height:
            self.last_line_height = word_shape[0]

        self.last_word_shape = word_shape

        return np.copy(self.offset)

    def get_padded_image(self, word, word_shape=None):
        if word_shape is None:
            word_shape = word.shape[0:2]

            if self.padded_image is None:
                self.padded_image = cv2.copyMakeBorder(word,
                                          self.offset[0],
                                          self.shape[0] - self.offset[0] - word_shape[0],
                                          self.offset[1],
                                          self.shape[1] - self.offset[1] - word_shape[1],
                                          cv2.BORDER_CONSTANT,
                                          (0, 0, 0, 0))
            else:
                self.padded_image[self.offset[0]:self.offset[0] + word_shape[0],
                                  self.offset[1]:self.offset[1] + word_shape[1]] = word

            return self.padded_image
        # else:
            # return cv2.copyMakeBorder(word,
                                      # self.offset[0] - word_shape[0],
                                      # self.shape[0] - self.offset[0] - word_shape[0],
                                      # self.offset[1] - word_shape[1],
                                      # self.shape[1] - self.offset[1] - word_shape[1],
                                      # cv2.BORDER_CONSTANT,
                                      # (0, 0, 0, 0))
