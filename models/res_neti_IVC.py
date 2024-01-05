
import h5py
import numpy as np
from PIL import Image
import io
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import pathlib
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
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import pickle
from itertools import zip_longest
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def training_data ():
    batch_size = 32
    input_height = 224
    input_width = 224

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'train_catdog',
    label_mode ='binary',
    validation_split=None,
    subset=None,
    seed=123,
    image_size=(input_height, input_width),
    batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'validation_catdog',
    label_mode ='binary',
    validation_split=None,
    subset=None,
    seed=123,
    image_size=(input_height, input_width),
    batch_size=batch_size)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds



train_ds, val_ds= training_data()



def resnet50 ():
    # The Convolutional Base of the Pre-Trained Model will be added as a Layer in this Model
    Conv_Base = ResNet50(include_top = False, weights = 'imagenet', input_shape = (224,224, 3))

    for layer in Conv_Base.layers[:-8]:
        layer.trainable = False

    model = Sequential()
    model.add(Conv_Base)
    model.add(Flatten())
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    model.summary()

    return model

model= resnet50 ()



def compile_fit_model():

    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = 'res_net_50-'+timestr #

    checkpoint_path = "checkpoints/"+name+"/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.system('mkdir {}'.format(checkpoint_dir))

    tensorboard_callback = TensorBoard(
    log_dir='tensorboard_logs/'+name,
    histogram_freq=1)

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = model.fit(train_ds, epochs=10, callbacks=[tensorboard_callback], validation_data=val_ds)
    model.save("ResNet50_model")

compile_fit_model()



def plot_loss():
    plt.plot(history.history['loss'],'r',label='training loss')
    plt.plot(history.history['val_loss'],label='validation loss')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
plot_loss()
    
def plot_acc():
    plt.plot(history.history['accuracy'],'r',label='training accuracy')
    plt.plot(history.history['val_accuracy'],label='validation accuracy')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
plot_acc()



def evaluate_model():
    score = model.evaluate(X_val, y_val)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
evaluate_model()

