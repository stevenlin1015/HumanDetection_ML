#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 17:31:04 2018

@author: stevenlin
"""

from PIL import Image



def compressImage(sourcePath, compressedPath):
    basewidth = 300
    image = Image.open(sourcePath)
    wpercent = (basewidth/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((basewidth,hsize), Image.ANTIALIAS) #單眼影像3:2
    image.save(compressedPath)

