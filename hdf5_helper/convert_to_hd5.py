from pathlib import Path
import numpy as np
import cv2
import os
import h5py
from itertools import zip_longest
import pathlib


def store_hdf5(image, idx):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file_h5 = h5py.File(f"/Users/salvatoreesposito/Downloads/Testing_dummy_data/data_not_processed_{idx}.hdf5", "w")

    # Create a dataset in the file
    dataset = file_h5.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    file_h5.close()

def load_image(filepath):

    img = cv2.imread(filepath)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return img

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

folder = "/Users/salvatoreesposito/Downloads/Testing_dummy_data/"
result = list(Path(folder).rglob("*.jpg"))
result = [str(r) for r in result]

for i, group in enumerate(grouper(result, 500)):
    hg=0
    images = np.array([load_image(img_path) for img_path in group if img_path is not None])
    store_hdf5(images, i)



def create_training_data_cleft(filepath, CATEGORIES):   
    DATADIR = '/Users/salvatoreesposito/Downloads/Testing_dummy_data/'
    CATEGORIES = ["normal, cleft"]
    X=[]
    y =[]
    for category in CATEGORIES:  # do normal and cleft
        folder = "/Users/salvatoreesposito/Downloads/Testing_dummy_data/"
        paths = list(Path(folder).rglob("*.hdf5"))
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=normal 1=cleft
        print(class_num)
        for file_name in paths:
            f = h5py.File(file_name, 'r')
            for key in f.keys():
                dset_read = f.get(key)
                dset_read_np = np.divide(np.array(dset_read),255)
                X.append(dset_read_np)
                y.append(class_num)
            f.close()
            print(file_name)
    return X,y
    create_training_data_cleft(filepath)