"""
Convert BDD100K label format to YOLOv5 format
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PATH = "/home/aferis/Datasets/BerkeleyDeepDrive/bdd100k/"

def object_category(c):
    ''' Switch case for the object classes defined in BDD-Dataset '''

    switcher = {
        'person': 0,
        'rider' : 1,
        'car'   : 2,
        'truck' : 3,
        'bus'   : 4,
        'train' : 5,
        'motor' : 6,
        'bike'  : 7,
        'traffic light' : 8,
        'traffic sign'  : 9
    }
    return switcher.get(c)

def process_data(data, data_type, w_img = 1280.00, h_img = 720.00):
    ''' Method responsible for the data conversion, which splits the JSON data into multiple txt-files '''

    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row["name"][:-4]               # Get the image name without ".jpg"
        labels = row["labels"]                      # Get the list of labels containing the category and the bounding box coordinates of each object
        yolo_data = []
        for object in labels:
            if 'box2d' in object:
                # Read the necessary values for each object delimited by a bounding box
                o = object_category(object.get('category'))
                x1 = object.get('box2d').get('x1')
                x2 = object.get('box2d').get('x2')
                y1 = object.get('box2d').get('y1')
                y2 = object.get('box2d').get('y2')

                # YOLO-format conversion with the normalization of the values
                x_center = x1 + (x2 - x1) / 2
                x_center /= w_img
                y_center = y1 + (y2 - y1) / 2
                y_center /= h_img
                w = (x2 - x1) / w_img
                h = (y2 - y1) / h_img
                yolo_data.append([o, x_center, y_center, w, h])
        yolo_data = np.array(yolo_data)
        np.savetxt(
            os.path.join(DATA_PATH, f"YOLOv5/{data_type}/{image_name}.txt"),
            yolo_data,
            fmt = ["%d", "%f", "%f", "%f", "%f"]     # Formatting the data types of each column
        )

if __name__ == "__main__":
    df_train = pd.read_json(os.path.join(DATA_PATH, "labels/bdd100k_labels_images_train.json"))
    df_val = pd.read_json(os.path.join(DATA_PATH, "labels/bdd100k_labels_images_val.json"))

    process_data(df_train, data_type="train")
    process_data(df_val, data_type="val")