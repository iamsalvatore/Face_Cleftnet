import h5py
import numpy as np
from PIL import Image
import io
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras import layers
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from keras.regularizers import l2
import pickle
from itertools import zip_longest
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
AUTOTUNE = tf.data.experimental.AUTOTUNE

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session =tf.compat.v1.InteractiveSession(config=config)
# gpus= tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)



def training_data ():
    batch_size = 3
    input_height = 512
    input_width = 512

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/Volumes/fmehenda-CLEFT/few_shot_unilateral_graded_images/',
    label_mode ='categorical',
    validation_split=None,
    subset=None,
    seed=123,
    image_size=(input_height, input_width),
    batch_size=batch_size)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return test_ds


test_ds = training_data()


def resize_and_augment(test_ds):

    # data_augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    # layers.experimental.preprocessing.RandomRotation(0.2)])

    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))

    return test_ds

test_ds = resize_and_augment(test_ds)



new_model = tf.keras.models.load_model('ResNet50_model')

# x = new_model.output

# x = new_model.layers[-1].output 

hidden = Dense(120, activation='relu',name='Dense_pre')(new_model.layers[-2].output)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax',name='Dense_final')(hidden)

# this is the model we will train
model = Model(inputs=new_model.input, outputs=predictions)
# Check its architecture
model.summary()

model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Evaluate the restored model
loss, acc = model.evaluate(test_ds, verbose=1)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(model.predict(test_ds).shape)