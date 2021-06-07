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
        if os.path.exists(self.DATASET_PATH):
