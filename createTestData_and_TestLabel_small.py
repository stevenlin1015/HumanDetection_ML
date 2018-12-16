#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2 as cv
import os
import numpy as np
import adjustPortraitImage_small

test_data = np.zeros((480, 199, 300, 3), dtype = np.float32)
test_label = np.zeros((480, 1), dtype = np.uint8)

def CreateTrainData_and_TrainLabel(testDataFolder):
    imageIndexCount = 0
    portraitCount = 0
    
    for root, dirs, files in os.walk(testDataFolder):
        for fileName in files:
            
            processingImage = cv.imread(testDataFolder + '/' + fileName, 1)            
            
            
            if processingImage.shape[0] < processingImage.shape[1]:
                for height in range(processingImage.shape[0]):
                    for width in range(processingImage.shape[1]):
                        test_data[imageIndexCount, height, width, 0] = processingImage[height, width, 0]
                        test_data[imageIndexCount, height, width, 1] = processingImage[height, width, 1]
                        test_data[imageIndexCount, height, width, 2] = processingImage[height, width, 2]
                        if imageIndexCount < 240:
                            test_label[imageIndexCount, 0] = 1 #tag "haspeople"
                        else:
                            test_label[imageIndexCount, 0] = 0 #tag "nopeople"

            elif processingImage.shape[0] > processingImage.shape[1]:
                processingImage = adjustPortraitImage_small.ExportCorrectPortraitImageSize(testDataFolder + '/' + fileName)
                portraitCount += 1
                for height in range(processingImage.shape[0]):
                    for width in range(processingImage.shape[1]):
                        test_data[imageIndexCount, height, width, 0] = processingImage[height, width, 0]
                        test_data[imageIndexCount, height, width, 1] = processingImage[height, width, 1]
                        test_data[imageIndexCount, height, width, 2] = processingImage[height, width, 2]
                        if imageIndexCount < 240:
                            test_label[imageIndexCount, 0] = 1 #tag "haspeople"
                        else:
                            test_label[imageIndexCount, 0] = 0 #tag "nopeople"
            
            print("完成第 " + str(imageIndexCount) + "張照片")
            imageIndexCount += 1
            
    np.save("/Users/stevenlin/Desktop/test_data.npy", test_data)
    print("test_data已儲存")
    np.save("/Users/stevenlin/Desktop/test_label.npy", test_label)
    print("test_label已儲存")
    
    print("共發現" + str(portraitCount) + " 筆非橫幅影像！")
    
CreateTrainData_and_TrainLabel("/Volumes/KINGSTON/ML_TrainingSet/Catagory_small/testData")