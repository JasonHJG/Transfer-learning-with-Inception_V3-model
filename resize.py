#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:13:52 2017

@author: jingang
"""

import PIL
from PIL import Image
import os
import shutil


path=os.path.dirname(os.path.abspath(__file__))
des=path+'/images/multi-label'
if os.path.isdir(des):
    shutil.rmtree(des)
else:   
    os.mkdir(path+'/images')     
    os.mkdir(des)

for i in range(2000):
    src=path+'/original/'+str(i+1)+".jpg"
    img=Image.open(src)
    img = img.resize((299, 299), PIL.Image.ANTIALIAS)
    img.save(des+'/'+str(i+1)+'.jpg')
    
  