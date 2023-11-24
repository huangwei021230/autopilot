import csv

from numpy.random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, Lambda
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
import pickle
from keras.optimizers import Adam

from TrainModel import loadFromPickle, augmentData

if __name__ == "__main__":
    model = load_model('Autopilot.h5')
    out = open("./data/Origin.csv", 'a', newline='')

    csv_writer = csv.writer(out)
    features, labels = loadFromPickle()
    features, labels = augmentData(features, labels)
    features, labels = shuffle(features, labels)
    print(labels)
    # train_x = features.reshape(features.shape[0], 40, 40, 1)
    #
    # activation_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    # activation_model_1 = Model(inputs=model.input, outputs=model.layers[-3].output)
    #
    # steering = model.predict(train_x, batch_size=16)
    # vec1 = activation_model.predict(train_x, batch_size=16)
    # vec2 = activation_model_1.predict(train_x, batch_size=16)
