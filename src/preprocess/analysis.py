import os
import pandas as pd
import numpy as np
import cv2 as cv


class Analysis:
    image = None
    partitioned_images = None

    def __init__(self, image_path):

        self.image = cv.imread(image_path)

        self.partition_image()

    def partition_image(self):
        partition_size = self.image.shape[0] // 4
        print(partition_size)

        # create four images from the input image

        image_array = []

        for i in range(4):
            init = partition_size * i
            image_array.append(
                self.image[init:init + partition_size, init:init + partition_size])

        self.partitioned_images = np.array(image_array)
        self.process_image()

    def process_image(self):
        image_one = self.partitioned_images[3]

        hsv = cv.cvtColor(image_one, cv.COLOR_BGR2HSV)

        sminusv = hsv[:, :, 1] - hsv[:, :, 2]

        cv.imshow("im", image_one)
        cv.waitKey(0)
