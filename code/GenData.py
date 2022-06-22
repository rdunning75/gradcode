import keras
import tensorflow as tf
import numpy as np
import os
import cv2

dir()

class GenData(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, dim=(100, 100), n_channels=1,
                 n_classes=1, time_step =1, shuffle=False):
        self.dim = dim
        self.time_step = time_step
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, path_to_data, time):
        # TO DO
        # generate a data set from the directories that I provide
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for root,folder_name, files in os.walk(path_to_data):
            if files.endswith('.tif'):
                X[i,] = cv2.imread(os.path.join(root,name))

                y[i] = self.labels[ID]
        
        X = tf.convert_to_tensor(X)
        X = tf.reshape(X,[self.batch_size,time,*self.dim,self.n_channels])
        return X, tf.keras.utils.to

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, item):

        # the item needs to be the directory of the data
        # we can set the labels to be the next image as this is what were trying to generat

        items = self.indexes[items*self.batch_size:(items+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X,y = self.__data_generation(list_IDs_temp)

        return X,y

