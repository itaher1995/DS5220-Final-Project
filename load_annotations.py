#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:08:19 2018

@author: forresthooton
"""

import json
import csv



def main():
    
    with open("/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/annotations_trainval2014/captions_train2014.json",'r') as f:
        valCaptionsJson = json.load(f)
    
    
    """
    captions_val2014.json structure
    read into valCaptionsJson (dict)
    
    Top level:
        - info (dict)
        - images (list)
        - licenses (list)
        - annotations [list]
    
    info:
        - dataset info
    
    images:
        -list of dicts
        - each dict in list contains:
            + license (int)
            + file_name (string) (I think the corresponding image files in the other folders)
            + coco_url (string)
            + height (int) (picture hieght)
            + width (int) (picture width)
            + date_captured (datetime)
            + flickr_url (string)
            + id (int) (I think corresponding to the id in annotations)
            
    licenses:
        - licenses for project
    
    annotations:
        - list of dicts
        - each dict in list contains:
            + image_id (int)
            + id (int)
            + caption (string)
    
    """
    
    with open("/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/train_annotations.csv", 'w') as csvfile:
        
        fieldnames = ['image_id', 'id', 'caption']
        
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        
        writer.writeheader()
        
        for row in valCaptionsJson['annotations']:
        
            writer.writerow(row)
      
            


if __name__ == "__main__":
    main()
