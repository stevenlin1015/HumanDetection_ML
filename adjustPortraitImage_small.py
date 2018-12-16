#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

def ExportCorrectPortraitImageSize(inputImagePath):
    newPortraitImage = np.zeros((199, 300, 3), dtype = np.uint8)
    
    inputImage = cv.imread(inputImagePath, 1)
    
    outputImage = cv.resize(inputImage, (132, 199))
    
    for height in range(199):
        for width in range(132):
            newPortraitImage[height, width + 84, 0] = outputImage[height, width, 0]
            newPortraitImage[height, width + 84, 1] = outputImage[height, width, 1]
            newPortraitImage[height, width + 84, 2] = outputImage[height, width, 2]
    
    
    return newPortraitImage
