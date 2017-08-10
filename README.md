# Transfer-learning-with-Inception_V3-model
Enable multi-label classification. The major change is that I replace the softmax loss to sigmoid loss.

# Environment 
The code is run on tensorflow

# Data set
The dataset is the MIMLimage (scene dataset) from NJU. The comprises of 5 classes with tree, mountain, desert, sea, sunset 

# Functions
## resize.py
resize the image to the size you want

## label_generator.py
read the original .mat file and find the label for each image and write in .txt file

## load_pd.py
read a .ckpt file and build the network and save it as a .pbfile

## retrain_2016.py 
the training process (I get this function from https://medium.com/towards-data-science/multi-label-image-classification-with-inception-net-cbb2ee538e30, the I add load_graph_from_local functionality so that you can load whatever model you want for the training

## label_image.py
to obtain the results for image you want to classify
