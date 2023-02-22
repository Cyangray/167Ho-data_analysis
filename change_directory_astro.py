#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:36:18 2023

@author: francesco
"""

import os

# folder path
dir_path = r'/home/francesco/Documents/164Dy-experiment/Python_normalization/Make_dataset/167Ho-saga-results'

# list to store files
res = []
# os.walk() returns subdirectories, file from current directory and 
# And follow next directory from subdirectory list recursively until last directory
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith("output.txt"):
            res.append(os.path.join(root, file))

for outputpath in res:
    folderpath = outputpath[:-10]
    print(folderpath)
    newfolderpath = os.path.join(folderpath,'jlmompy')
    os.mkdir(newfolderpath)
    os.rename(outputpath, newfolderpath + '/output.txt')
    os.rename(folderpath + '/astrorate.g', newfolderpath + '/astrorate.g')
    os.rename(folderpath + '/astrorate.tot', newfolderpath + '/astrorate.tot')
