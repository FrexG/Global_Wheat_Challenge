import os
import pandas as pd
import numpy as np
# convert wheat image dataset into the format
# required by YoLov4 architecture


class Convert2YoLo:
    # initialize dataset path
    DATASET_PATH = "/home/frexg/Documents/Global_Wheat_Challenge/train/train"
    LABELS_PATH = "/home/frexg/Documents/Global_Wheat_Challenge/train/train.csv"

    YOLO_TRAIN_PATH = "/home/frexg/Documents/Global_Wheat_Challenge/yolo/train"
    YOLO_TEST_PATH = "/home/frexg/Documents/Global_Wheat_Challenge/yolo/test"

    def __init__(self):
        # If dataset exists
        try:
            labels_df = pd.read_csv(self.LABELS_PATH)

            self.__process_labels(labels_df)

        except FileNotFoundError:
            print(f"The path '{self.LABELS_PATH}'is INVALID")

    def __process_labels(self, dataframe):
        # convert the dataframe into numpy array
        dataframe_array = np.array(dataframe)

        for i, image in enumerate(dataframe_array):
            # get image name
            image_name = image[0]
            # extract roi
            region_of_interests = image[1]
            domain = image[2]
            # Split by semicolon and space
            split_roi = region_of_interests.split(';')
            split_roi = [r.split(" ") for r in split_roi]
            # cast string into integer
            split_roi_int = [list(map(int, r)) for r in split_roi]

            print(split_roi_int)

            break


if __name__ == "__main__":

    converter = Convert2YoLo()
