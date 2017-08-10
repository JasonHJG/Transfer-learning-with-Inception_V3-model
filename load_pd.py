#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:05:31 2017

@author: jingang
"""



import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename ="/home/jingang/Downloads/inception_V3/inception-v3-retrain.pb"
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        #InceptionV3/InceptionV3/Mixed_7c/Branch_3/AvgPool_0a_3x3/AvgPool
        #InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0
        current_tensor = tf.import_graph_def(graph_def, name='', return_elements=["InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0"])
        print current_tensor
        current_tensor = tf.import_graph_def(graph_def, name='', return_elements=["DecodeJpeg/contents:0"])
        
        print current_tensor
        current_tensor = tf.import_graph_def(graph_def, name='', return_elements=['ResizeBicubic:0'])
        print current_tensor
       
        file = open('/home/jingang/Downloads/inception_V3/model_2016.txt','w') 
        for node in graph_def.node:
            file.write(node.name+'\n')
            
            

'''  

with tf.Session() as sess:
    model_filename ="/home/jingang/Downloads/voxelCloud/classify_image_graph_def.pb"
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        
        current_tensor = tf.import_graph_def(graph_def, name='', return_elements=["pool_3/_reshape:0"])
        print current_tensor
        current_tensor = tf.import_graph_def(graph_def, name='', return_elements=["DecodeJpeg/contents:0"])
        print tf.shape(current_tensor)
        print current_tensor
        current_tensor = tf.import_graph_def(graph_def, name='', return_elements=['ResizeBilinear:0'])
        print current_tensor
                
        
        file = open('/home/jingang/Downloads/voxelCloud/model_2015.txt','w') 
        for node in graph_def.node:
            file.write(node.name+'\n')

'''    
LOGDIR='/home/jingang/Downloads/inception_V3/logs2/test/1'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
