#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 16:09:51 2018

@author: forresthooton
"""

import pandas as pd
import fnmatch
import os


def main():
    annotations = pd.read_csv("/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/annotations.csv")
    print(annotations.head())
    
    z = len(annotations)
    print(z)

    
    dirpath = "/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/train2014/"
    x = len(fnmatch.filter(os.listdir(dirpath), '*.jpg'))
    print(x)
    
    dirpath = "/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/val2014/"
    y = len(fnmatch.filter(os.listdir(dirpath), '*.jpg'))
    print(y)
    
    dirpath = "/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/test2014"
    w = len(fnmatch.filter(os.listdir(dirpath), '*.jpg'))
    print(w)
    
    print(w+ x + y)
    
if __name__ == "__main__":
    main()


