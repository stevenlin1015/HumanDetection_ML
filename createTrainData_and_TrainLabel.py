#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2 as cv
import os
import numpy as np
import adjustPortraitImage

train_data = np.zeros((24000, 199, 300, 1), dtype = np.float32)
train_label = np.zeros((24000, 1), dtype = np.uint8)

def CreateTrainData_and_TrainLabel(trainDataFolder):
    imageIndexCount = 0
    
    
    for root, dirs, files in os.walk(trainDataFolder):
        for fileName in files:
            
            processingImage = cv.imread(trainDataFolder + '/' + fileName, 0)            
            
            
            if processingImage.shape[0] < processingImage.shape[1]:
                for height in range(processingImage.shape[0]):
                    for width in range(processingImage.shape[1]):
                        train_data[imageIndexCount, height, width, 0] = processingImage[height, width]
                        if imageIndexCount < 12000:
                            train_label[imageIndexCount, 0] = 1 #tag "haspeople"
                        else:
                            train_label[imageIndexCount, 0] = 0 #tag "nopeople"
                
            elif processingImage.shape[0] > processingImage.shape[1]:
                processingImage = adjustPortraitImage.ExportCorrectPortraitImageSize(trainDataFolder + '/' + fileName)
                
                for height in range(processingImage.shape[0]):
                    for width in range(processingImage.shape[1]):
                        train_data[imageIndexCount, height, width, 0] = processingImage[height, width]
                        if imageIndexCount < 12000:
                            train_label[imageIndexCount, 0] = 1 #tag "haspeople"
                        else:
                            train_label[imageIndexCount, 0] = 0 #tag "nopeople"
            print("fileName : " + fileName + " , tag : " + str(train_label[imageIndexCount, 0]))
            
            print("完成第 " + str(imageIndexCount) + "張照片")
            imageIndexCount += 1
    '''    
    np.save("/Users/stevenlin/Desktop/train_data.npy", train_data)
    print("train_data已儲存")
    np.save("/Users/stevenlin/Desktop/train_label.npy", train_label)
    print("train_label已儲存")
    '''
CreateTrainData_and_TrainLabel("/Volumes/KINGSTON/ML_TrainingSet/Catagory/trainData")