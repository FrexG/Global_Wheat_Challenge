import os
import pandas as pd
import numpy as np
from shutil import copy2
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

            self.process_labels(labels_df)

        except FileNotFoundError:
            print(f"The path '{self.LABELS_PATH}'is INVALID")

    def process_labels(self, dataframe):
        # convert the dataframe into numpy array
        dataframe_array = np.array(dataframe)

        for i, image in enumerate(dataframe_array):
            print(f"Processing {i} of {len(dataframe_array)}")
            # get image name
            image_name = image[0]
            # extract roi
            region_of_interests = image[1]
            domain = image[2]

            # only images with visible wheat heads
            if region_of_interests != "no_box":
                # Split by semicolon and space
                split_roi = region_of_interests.split(';')
                split_roi = [r.split(" ") for r in split_roi]
                # cast string into integer
                split_roi_int = [list(map(int, r)) for r in split_roi]

                # find the yolo bounding box centers
                yoloformat = self.__find_yolo_centers(split_roi_int)
                # save to a new folder
                self.__writeToFile(image_name, yoloformat)

    def __find_yolo_centers(self, region_of_interests):
        __ = []

        for box in region_of_interests:
            x_center = (box[0] + box[2]) // 2
            y_center = (box[1] + box[3]) // 2

            __.append([0, x_center, y_center, box[2], box[3]])

        return __

    def __writeToFile(self, image_name, yolo_labels):
        labels = np.array(yolo_labels)
        try:
            copy2(os.path.join(self.DATASET_PATH,
                               f"{image_name}.png"), self.YOLO_TRAIN_PATH)

        except Exception:
            print(Exception)
        finally:
            np.savetxt(os.path.join(self.YOLO_TRAIN_PATH, f"{image_name}.txt"), labels, fmt=[
                "%d", "%d", "%d", "%d", "%d"])


if __name__ == "__main__":

    converter = Convert2YoLo()
