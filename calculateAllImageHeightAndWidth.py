#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import cv2 as cv
import numpy as np

imageSize = np.zeros((2,1), dtype = np.int16)

def CalculateAllImageHeightAndWidth(path):
    global imageSize
    size = [0, 0, 0]
    imageSize[0,0] = 1000
    imageSize[1,0] = 1000
    
    for root, dirs, files in os.walk(path):
        for fileName in files:
            img = cv.imread(path + '/' + fileName, 1)
            print("An image size is " + str(imageSize[0,0]) + " x " +str(imageSize[1,0]))
            size = img.shape
            if (imageSize[0,0] > size[0]) and (imageSize[1,0] > size[1]):
                imageSize[0,0] = size[0]
                imageSize[1,0] = size[1]
                
                print("error occur! Now image size is " + str(imageSize[0,0]) + " x " + str(imageSize[1,0]))
            
            '''
            cv.imshow("img", img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            '''
    

CalculateAllImageHeightAndWidth("/Volumes/KINGSTON/ML_TrainingSet/trainingFolder/nopeople_compress")