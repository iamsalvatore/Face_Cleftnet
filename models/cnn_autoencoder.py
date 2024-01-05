import numpy as np
import io
from keras.utils import np_utils
from os import listdir
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
import pickle
import dlib
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import os
import re
from scipy import ndimage, misc
from matplotlib import pyplot
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from itertools import zip_longest
import keras
from keras import layers


def training_data(data_path,data_dir):

    categories=os.listdir(data_path)
    labels=[i for i in range(len(categories))]
    label_dict=dict(zip(categories,labels)) #empty dictionary
    data_train=[]
    target_train=[]
    img_size=512
    for category in categories:
        path = os.path.join(data_dir,category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                width = 512
                height = 512
                dim = (width, height)
                img = cv2.resize(img_array, dim, interpolation = cv2.INTER_AREA)  
                new_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)          
                data_train.append(new_img)
                target_train.append(label_dict[category])
            except Exception as e:
                print(str(e))
    data_train=np.array(data_train)
    target_train=np.array(target_train)

    return data_train, target_train



def reshape(data_train, target_train):

    X_train, X_test, y_train, y_test = train_test_split(data_train, target_train)
    X_train =  X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_train = np.reshape(X_train, (len(X_train), 512, 512, 3))
    X_test = np.reshape(X_test, (len(X_test), 512, 512, 3))

    return X_train,X_test,y_train,y_test



def cnn_autoencoder (pre_trained_weights=None):

    input_img = keras.Input(shape=(512, 512, 3))

    # Encoder
    l1 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)
    l2 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)
    l3 = MaxPooling2D(padding='same')(l2)
    l3 = Dropout(0.3)(l3)
    l4 = Conv2D(128, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l3)
    l5 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)
    l6 = MaxPooling2D(padding='same')(l5)
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l6)

    # Decoder
    l8 = UpSampling2D()(l7)
    l9 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l9)
    l11 = add([l5, l10])
    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l12)
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l13)
    l15 = add([l14, l2])

    #chan = 3, for RGB
    decoded = Conv2D(3, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l15)


    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    if pre_trained_weights:
        autoencoder.load(pre_trained_weights)


    autoencoder.summary()

    return autoencoder




def compile_fit_model(X_train,X_test):

    autoencoder.fit(X_train,X_train,
                    epochs=10,
                    batch_size=3,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    )


def show_pred(autoencoder):

    decoded_imgs = autoencoder.predict(X_test)
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(X_test[i].reshape(512, 512))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(512, 512))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


    def main():
        data_path='/Users/salvatoreesposito/Downloads/Dummy_faces_data/'
        data_dir = '/Users/salvatoreesposito/Downloads/Dummy_faces_data/'
        data_train, target_train=training_data()
        X_train,X_test,y_train,y_test=reshape(data_train, target_train)
        autoencoder = cnn_autoencoder(pre_trained_weights=None)
        X_train,X_test=compile_fit_model()

