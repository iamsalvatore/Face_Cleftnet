import h5py
import numpy as np
from PIL import Image
import pandas as pd
import io
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import pathlib
from tensorflow import keras
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
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.layers.experimental import preprocessing
import pickle
from itertools import zip_longest
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
AUTOTUNE = tf.data.experimental.AUTOTUNE
import cbam

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session =tf.compat.v1.InteractiveSession(config=config)
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



def training_data ():

    batch_size = 1
    input_height = 299
    input_width = 299


    train_directory = '/disk/scratch/dataset/dataframe.csv'
    valid_directory =  '/disk/scratch/dataset/dataframe_validate.csv'
    test_directory = '/disk/scratch/dataset/dataframe_test.csv'

    df_train = pd.read_csv(train_directory)
    df_validate = pd.read_csv(valid_directory)
    df_test = pd.read_csv(test_directory)

    # All images will be rescaled by 1./255, 
    # and random augmentation are added to the training generator
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    # Note that validation and test generators must not use augmentations!
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        # This is the target directory
        dataframe=df_train,
        directory=None,
        x_col="Path",
        y_col="Score",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="other",
        target_size=(input_height,input_width))


    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=df_validate,
        directory=None,
        x_col="Path",
        y_col="Score",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="other",
        target_size=(input_height,input_width))


    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=None,
        x_col="Path",
        y_col="Score",
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode="other",
        target_size=(input_height,input_width))

  

    return train_generator, validation_generator, test_generator


train_generator, validation_generator,test_generator = training_data()



def resize_and_augment(train_ds, val_ds,test_ds):
    # data_augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    # layers.experimental.preprocessing.RandomRotation(0.2)])

    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))

    return train_ds, val_ds,test_ds

train_ds,val_ds,test_ds = resize_and_augment(train_ds, val_ds, test_ds)


def cbam_model():

    # loss_used = 'mean_squared_error'


    inputs = keras.Input(shape=(299, 299, 3))



    model = resnet(input_shape,
           num_classes,
           normalization='batchnorm',
           activation='relu',
           use_cbam = True,
           name = 'ResNet')

    
    x = model.output
    
    x = layers.Flatten()(x)

    x = Dense(1024, activation='relu')(x)

    x = layers.Dropout(0.5)(x)

    # and a logistic layer -- let's say we have 200 classes
    x = layers.Dense  (1, activation='linear')(x)    
    # this is the model we will train
    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(optimizer = RMSprop(lr=.0001), loss = 'mean_squared_error', metrics = ['accuracy'])


    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in model.layers:
    #     layer.trainable = False

    model.summary()

    return model

model = InceptionV3_model ()


def compile_fit_model(train_generator, validation_generator,model):

   
        # Fit model using the generator method.
    history= model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        verbose=1)
    
    # history = model.fit(train_ds, epochs=30, validation_data=val_ds, batch_size=2)
    
    model.save("InceptionV3_continous_model")
    
    return history

history = compile_fit_model(train_generator, validation_generator,model)



def plot_loss():
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.show()
    plt.savefig("Inception_continous_preprocessing_categorical_loss_inception.svg")
plot_loss()
    

def plot_acc():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 8))
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig("Inception_continous_preprocessing_categorical_acc_inception.svg")
plot_acc()



def evaluate_model(test_generator):

    loss, acc = model.evaluate(test_generator)
    print("Accuracy", acc)
    print("Loss",loss)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])
evaluate_model(test_generator)
