import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation
from keras.initializers import VarianceScaling
from keras.optimizers import Adam
from keras.utils import np_utils, multi_gpu_model
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.utils import class_weight
np.set_printoptions(suppress=True)

trainD = np.load("/home/hsiehch/project/human_detection/train_data.npy")
trainL = np.load("/home/hsiehch/project/human_detection/train_label.npy")
testD = np.load("/home/hsiehch/project/human_detection/test_data.npy")
testL = np.load("/home/hsiehch/project/human_detection/test_label.npy")


trainData = trainD / 255
testData = testD / 255


trainData_normalize = trainD.reshape(trainData.shape[0], trainData.shape[1], trainData.shape[2], 3).astype('uint8')
trainLabel_onehot = np_utils.to_categorical(trainL, 2)
testData_normalize = testD.reshape(testData.shape[0], testData.shape[1], testData.shape[2], 3).astype('uint8')
testLabel_onehot = np_utils.to_categorical(testL, 2)
print('Train Data:', trainData_normalize.shape)
print('Train Label: ', trainLabel_onehot.shape)
print('Test Data: ', testData_normalize.shape)
print('Test Label: ', testLabel_onehot.shape)

#建立CNN模型
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (2, 2), input_shape = (trainData_normalize.shape[1], trainData_normalize.shape[2], 3), data_format = "channels_last"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format = "channels_last"))

model.add(Conv2D(filters = 32, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format = "channels_last"))

model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (3, 3), data_format = "channels_last"))

model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (3, 3), data_format = "channels_last"))

#建立分類模型
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = "softmax"))

print(model.summary())


adam = Adam(lr = 0.001)
model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics=['accuracy'])

early_stop = EarlyStopping(patience=20)

train_history_1 = model.fit(x = trainData_normalize,
                            y = trainLabel_onehot,
                            epochs=20,
                            validation_data=(testData_normalize, testLabel_onehot),
                            validation_split=0.2,
                            batch_size=100, 
                            verbose=1
                           )

evaluation = model.evaluate(x = testData_normalize, y = testLabel_onehot)
print('Loss: {:.3f}, Accuracy: {:.3f}'.format(evaluation[0], evaluation[1]))


model.save("human_detection_model.h5")

print('Finish training!')

import pylab as plt
def history_display(hist, train, validation):
    plt.plot(hist.history[train])
    plt.plot(hist.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show
    
def show_plot(flag, hist):
    if flag == 'acc':
        history_display(hist, 'acc', 'val_acc')
    elif flag == 'loss':
        history_display(hist, 'loss', 'val_loss')
    else:
        print('Invalid!')