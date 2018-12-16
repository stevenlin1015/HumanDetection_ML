#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

def ExportCorrectPortraitImageSize(inputImagePath):
    newPortraitImage = np.zeros((199, 300, 1), dtype = np.uint8)
    
    inputImage = cv.imread(inputImagePath, 0)
    
    outputImage = cv.resize(inputImage, (132, 199))
    
    for height in range(199):
        for width in range(132):
            newPortraitImage[height, width + 84] = outputImage[height, width]
    
    
    return newPortraitImage
