#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2 as cv
import os

def RotateImage(input_ImageFolderPath, inputImageName, rotateAngle):
    sourceImage = cv.imread(input_ImageFolderPath + '/' + inputImageName, 1)
    height, width, channels = sourceImage.shape

    '''
    parameters: 旋轉中心, 旋轉角度, 縮放比例
    '''
    M = cv.getRotationMatrix2D((width/2, height/2), int(rotateAngle), 1)
    '''
    parameters: 來源影像(輸入), cv.getRotationMatrix2D函數, 輸出影像大小
    '''
    outputRotatedImage = cv.warpAffine(sourceImage, M, (width,height))
    '''
    plt.subplot(121)
    plt.imshow(sourceImage)
    plt.subplot(122)
    plt.imshow(outputRotatedImage)
    '''
    return outputRotatedImage
    


def RetrieveAllImageName(angles, input_ImageFolderPath, output_ImageFolderPath):
    imageIndexCount = 0
    
    for root, dirs, files in os.walk(input_ImageFolderPath):
        for fileName in files:
            
            for rotateAngle in angles:
                outputRotetedImage = RotateImage(input_ImageFolderPath, fileName, str(rotateAngle))
                cv.imwrite(str(output_ImageFolderPath) + '/' + str(imageIndexCount) + '_' + str(rotateAngle) + '.JPG',outputRotetedImage)
            
            imageIndexCount += 1
            print(imageIndexCount)
        
angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
RetrieveAllImageName(
        angles,
        '/Volumes/KINGSTON/ML_TrainingSet/trainingFolder/nopeople_compress',
        '/Volumes/KINGSTON/ML_TrainingSet/trainingFolder/nopeople_compress_dataaug'
                     )

