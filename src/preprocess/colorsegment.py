"""
    ####Image Segmentation class###
    @author: Fraol Gelana
    @Institute:Artificial Intelligence Center
    @Date:December,2020
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


class ImageSegment:
    image = None
    extracted = None

    def __init__(self, image):
        self.image = image
        self.currentMeanArray = []      # Empty array to hold the calculated Mean,
        self.currentVarianceArray = []  # Variance and
        self.areaUnderArray = []        # area under the mean-variance line
        self.remove_background(image)  # read image

    def __setExtracted(self, bg_extracted):
        self.extracted = bg_extracted

    def getExtracted(self):
        return self.extracted

    def remove_background(self, leaf_image):

        # Convert blured Image from BGR to HSV
        hsv_leaf = cv.cvtColor(leaf_image, cv.COLOR_BGR2HSV)

        SV_channel = hsv_leaf.copy()

        SV_channel[:, :, 0] = np.zeros(
            (SV_channel.shape[0], SV_channel.shape[1]))  # Set the 'H' channel to Zero

        # Create a binary mask from the SV Channel

        mask = cv.inRange(SV_channel, (0, 0, 100), (0, 90, 255))
        # Invert mask, White areas represent green components and black the background
        # mask = cv.bitwise_not(mask)

        # perform bitwise_and between mask and hsv image

        background_extracted = cv.bitwise_and(hsv_leaf, hsv_leaf, mask=mask)

        self.image_correction(background_extracted, leaf_image)

    def image_correction(self, hsv_image, leaf_image):
        # Histogram Equalize the 'V' channel of the hsv image
        hsv_image_equ = hsv_image.copy()

        rgb_image_equ = cv.cvtColor(hsv_image_equ, cv.COLOR_HSV2RGB)

        self.color_segment(hsv_image_equ, rgb_image_equ, leaf_image)

    def color_segment(self, hsv_space, rgb_space, leaf_image, lowerB=(0, 0, 0), upperB=(255, 255, 255)):
        # extracted in the HSV color space

        # create binary mask using the bounds
        mask = cv.inRange(hsv_space, lowerB, upperB)
        # mask = cv.bitwise_not(mask)

        # bitwise_and mask and rgb image
        output_hsv = cv.bitwise_and(hsv_space, hsv_space, mask=mask)
        output_rgb = cv.bitwise_and(rgb_space, rgb_space, mask=mask)

        nonZeroIntentsity = output_hsv[:, :, 0].copy()

        # Extract intensity values between [3,65]
        # This step ensures the black(intensity 0 - 2) pixel values
        # from the mean calculation.
        nonZeroIntentsity = nonZeroIntentsity[nonZeroIntentsity > 3]
        nonZeroIntentsity = nonZeroIntentsity[nonZeroIntentsity < 255]

        l = self.findLowerBound(nonZeroIntentsity)
        print(f'Lower Bound: {l}')
        # Update the lower bound value of the 'H' channel accordingly
        # (H,S,V)
        new_lowerB = (l, 0, 0)

        mask = cv.inRange(hsv_space, new_lowerB, upperB)
        mask = cv.bitwise_not(mask)

        # bitwise_and mask and rgb image
        o_rgb = cv.bitwise_and(output_rgb, output_rgb, mask=mask)
        o_hsv = cv.bitwise_and(hsv_space, hsv_space, mask=mask)

        self.__setExtracted(o_rgb)

        #self.thresh_mask(o_rgb, leaf_image)

    def findLowerBound(self, intensityArray):

        nonZeroIntentsity = intensityArray

        # Find the mean and variance of the nonzero
        # intensity array
        mean = int(np.mean(nonZeroIntentsity))
        variance = int(np.var(nonZeroIntentsity))

        # Add the calculated mean and variance into
        # currentMeanArray and currentVarianceArray,respectively
        self.currentMeanArray.append(mean)
        self.currentVarianceArray.append(variance)

        # Update the nonZeroIntensity array,according to the current
        # mean value
        nonZeroIntentsity = nonZeroIntentsity[nonZeroIntentsity <= mean]
        # Calculate the area of the right-triangle formed by the value of the Variance
        # and the value of the Mean
        #           |\
        #           | \
        #   Variance|  \
        #           |   \
        #           |____\
        #            Mean

        self.areaUnderArray.append(
            (self.currentMeanArray[-1] * self.currentVarianceArray[-1]) / 2)

        print(f'Area: {self.areaUnderArray}')
        print(f'Mean: {self.currentMeanArray}')
        print(f'Var : {self.currentVarianceArray}')
        print(
            f"----------------Iteration[{len(self.areaUnderArray)}]-----------------")

        if len(self.areaUnderArray) >= 2 and self.areaUnderArray[-2] >= self.areaUnderArray[-1]:

            return self.currentMeanArray[-2]
        else:
            lowerBound = self.findLowerBound(nonZeroIntentsity)
            return lowerBound

    def thresh_mask(self, out, leaf_image):
        mask = cv.cvtColor(out, cv.COLOR_RGB2GRAY)
        # Calculate the otsu threshold
        ret, thresh = cv.threshold(
            mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Morphological close operation
        kernel = np.ones((9, 9))

        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

        self.__setExtracted(thresh)

        #self.find_contours(thresh, out, leaf_image, leafArea)

    def find_contours(self, mask, img, leaf_image, leafArea):
        # Find the contours of the segmented disease spots
        leaf_image = cv.cvtColor(leaf_image, cv.COLOR_BGR2RGB)
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
