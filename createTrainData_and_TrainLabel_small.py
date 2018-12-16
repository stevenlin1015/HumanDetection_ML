#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2 as cv
import os
import numpy as np
import adjustPortraitImage_small

train_data = np.zeros((1920, 199, 300, 3), dtype = np.float32)
train_label = np.zeros((1920, 1), dtype = np.uint8)

def CreateTrainData_and_TrainLabel(trainDataFolder):
    imageIndexCount = 0
    portraitCount = 0
    
    for root, dirs, files in os.walk(trainDataFolder):
        for fileName in files:
            
            processingImage = cv.imread(trainDataFolder + '/' + fileName, 1)            
            
            
            if processingImage.shape[0] < processingImage.shape[1]:
                for height in range(processingImage.shape[0]):
                    for width in range(processingImage.shape[1]):
                        train_data[imageIndexCount, height, width, 0] = processingImage[height, width, 0]
                        train_data[imageIndexCount, height, width, 1] = processingImage[height, width, 1]
                        train_data[imageIndexCount, height, width, 2] = processingImage[height, width, 2]
                        if imageIndexCount < 960:
                            train_label[imageIndexCount, 0] = 1 #tag "haspeople"
                        else:
                            train_label[imageIndexCount, 0] = 0 #tag "nopeople"
                
            elif processingImage.shape[0] > processingImage.shape[1]:
                processingImage = adjustPortraitImage_small.ExportCorrectPortraitImageSize(trainDataFolder + '/' + fileName)
                portraitCount += 1
                for height in range(processingImage.shape[0]):
                    for width in range(processingImage.shape[1]):
                        train_data[imageIndexCount, height, width, 0] = processingImage[height, width, 0]
                        train_data[imageIndexCount, height, width, 1] = processingImage[height, width, 1]
                        train_data[imageIndexCount, height, width, 2] = processingImage[height, width, 2]
                        if imageIndexCount < 960:
                            train_label[imageIndexCount, 0] = 1 #tag "haspeople"
                        else:
                            train_label[imageIndexCount, 0] = 0 #tag "nopeople"
            
            print("完成第 " + str(imageIndexCount) + "張照片")
            imageIndexCount += 1
      
    np.save("/Users/stevenlin/Desktop/train_data.npy", train_data)
    print("train_data已儲存")
    np.save("/Users/stevenlin/Desktop/train_label.npy", train_label)
    print("train_label已儲存")
    
    print("共發現" + str(portraitCount) + " 筆非橫幅影像！")
    
CreateTrainData_and_TrainLabel("/Volumes/KINGSTON/ML_TrainingSet/Catagory_small/trainData")