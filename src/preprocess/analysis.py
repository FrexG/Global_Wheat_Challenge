import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from .colorsegment import ImageSegment


class Analysis:
    image = None

    def __init__(self, image_path):

        self.image = cv.imread(image_path)

        self.process_image()

    def process_image(self):

        bg = ImageSegment(self.image).getExtracted()

        image_one = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        #bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image_one)
        axs[1].imshow(bg[:, :, 2], cmap="gray")

        plt.show()
