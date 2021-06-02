# main entry of the program
# ----------------------------##
# Accessing and manipulation of the ##
# dataset csv file##
import os
import pandas as pd
import numpy as np
import cv2 as cv

# get current working directory

dataset_path = "/home/frexg/Documents/Global_Wheat_Challenge/train/train"
csv_path = "/home/frexg/Documents/Global_Wheat_Challenge/train/train.csv"

if os.path.exists(csv_path):
    # create a dataframe form the csv file
    dataframe = pd.read_csv(csv_path)

    image_name = f'{dataframe.iloc[100, 0]}.png'
    regions = dataframe.iloc[100, 1]
    regions_array = regions.split(';')

    regions_array = [r.split(" ") for r in regions_array]

    regions_np = np.array(regions_array)

    # print(int(regions_array[0][0]))
    print(regions_np[0])

    image = cv.imread(os.path.join(dataset_path, image_name))

    for r in regions_np:
        x = int(r[0])
        y = int(r[1])
        w = int(r[2])
        h = int(r[3])

        cv.rectangle(image, (x, y), (w, h), (0, 0, 255), 3)

    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


else:
    print(f'The path "{csv_path}" does not exist')
