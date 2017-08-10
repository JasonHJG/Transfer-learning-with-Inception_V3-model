#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:11:48 2017

@author: jingang
"""
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_arg_scope
tf.reset_default_graph()

from tensorflow.python.framework import graph_util





#import tensorflow.contrib.slim.nets.inception.inception_v3_arg_scope as inception_v3_arg_scope
#from tensorflow.contrib.slim.nets.inception import inception_v3

checkpoint_file = '/home/jingang/Downloads/inception_V3/inception_v3.ckpt'




with tf.Session() as sess:
    decode_jpeg = tf.image.decode_jpeg("jpeginput", channels=3)
    if decode_jpeg.dtype != tf.float32:
        decode_jpeg = tf.image.convert_image_dtype(decode_jpeg, dtype=tf.float32)
    image_ = tf.expand_dims(decode_jpeg, 0)
    image = tf.image.resize_bicubic(image_, [299, 299], align_corners=True)
    scaled_input_tensor = tf.scalar_mul((1.0/255), image)
    scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
    scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
    
    arg_scope = inception_v3.inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v3.inception_v3(inputs=scaled_input_tensor, is_training=False, num_classes=1001)
    save = tf.train.Saver()
    save.restore(sess, checkpoint_file)
   
    output_filename = '/home/jingang/Downloads/inception_V3/inception-v3-retrain.pb'
    graph = sess.graph
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ['InceptionV3/Predictions/Reshape_1'])
    with gfile.FastGFile(output_filename, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    
'''
decode_jpeg = tf.image.decode_jpeg("jpeginput", channels=3)
if decode_jpeg.dtype != tf.float32:
  decode_jpeg = tf.image.convert_image_dtype(decode_jpeg, dtype=tf.float32)
image_ = tf.expand_dims(decode_jpeg, 0)
image = tf.image.resize_bicubic(image_, [299, 299], align_corners=True)
scaled_input_tensor = tf.scalar_mul((1.0/255), image)
scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

# loading the inception graph
arg_scope = inception_v3_arg_scope()
with slim.arg_scope(arg_scope):
  logits, end_points = inception_v3.inception_v3(inputs=scaled_input_tensor, is_training=False, num_classes=1001)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, checkpoint_file)


  


inference_graph=sess.graph
graph_def=inference_graph.as_graph_def()
with gfile.FastGFile('/home/jingang/Downloads/inception_V3/inceptionv3.pb', 'wb') as f:
    #f.write(output_graph_def.SerializeToString())

    f.write(graph_def.SerializeToString())
    
'''    