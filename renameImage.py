#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import compressImage


def renameImage(path, outputRootPath):
    for root, dirs, files in os.walk(path):
        count = 240
        for fileName in files:
            print os.path.join(outputRootPath, str(count) + ".JPG")
            #compressImage.compressImage(os.path.join(root, fileName), os.path.join(outputRootPath, str(count) + "_comp.JPG"))
            os.rename(os.path.join(path, fileName), os.path.join(outputRootPath, str(count) + ".JPG"))
            count += 1
            if count == 480:
                return
    print("Done! count = " + str(count))
            
renameImage("/Volumes/KINGSTON/ML_TrainingSet/nopeople_compress", "/Volumes/KINGSTON/ML_TrainingSet/Catagory_small/testData")
