#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:57:55 2017

@author: jingang
"""

import scipy.io as spio
import os
import shutil


# read the correct class and label
mat = spio.loadmat('miml data.mat', squeeze_me=True)

label = mat['targets'] # targets
length, width = label.shape

dir="/home/jingang/Downloads/inception/image_labels_dir"

if os.path.isdir(dir):
    shutil.rmtree(dir)
else:
    os.mkdir(dir)

class_name=['desert','mountain','sea','sunset','tree']
for i in range(1,width+1):
    dirname=dir+"/"+str(i)+".jpg.txt"
    if os.path.isfile(dirname):
        os.remove(dirname)
    f= open(dirname,'w')
    string=[]
    for j in range(5):
        if label[j,i-1]==1:
            
            string.append(class_name[j])
            
    for word in string:
         f.write(word+'\n')
    f.close()     

    
    
    
    