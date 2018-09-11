# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:07:56 2018

@author: shrey
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

num_train_each_letter=60 # number of training examples to take for each letter
num_test_each_letter=2 # number of testing examples to take for each letter

num_train=num_train_each_letter*62 # 62 are the number of classes: 0-9,a-z,A-Z
num_test=num_test_each_letter*62

image_len_flat=128*128*3  # After converting to grayscale

train_path="./English/Fnt"

directory=os.listdir(train_path)

train_set_x=np.zeros([num_train,image_len_flat])
train_set_y=np.zeros([num_train,62]) # 62 are the number of classes: 0-9,a-z,A-Z
test_set_x=np.zeros([num_test,image_len_flat])
test_set_y=np.zeros([num_test,62])

train_index=0
test_index=0
for folder in directory:
    item_count=0
    item_count_test=0
    folder_path=train_path+'/'+folder
    cls=int(filter(str.isdigit, folder)) # Get the class number from the folder name
    for item in os.listdir(folder_path):
        image=cv2.imread(folder_path+'/'+item)
        arr=np.array(image)
        f_arr=arr.flatten()
        if item_count>=num_train_each_letter:
            test_set_x[test_index]=f_arr
            test_set_y[test_index][cls-1]=1
            test_index+=1
            item_count_test+=1
            if item_count_test>=num_test_each_letter:
                break
        else:    
            train_set_x[train_index]=f_arr
            train_set_y[train_index][cls-1]=1
            train_index+=1
            item_count+=1 
'''
# Visualize the training set. Uncomment this block and run the following script:
image_num=70       # Change image number till num_train   
cv2.imshow("img",train_set_x[image_num].reshape([128,128,3]))
cv2.waitKey(0)
cv2.destroyAllWindows() 
'''                  
            
# Change datatype and normalize training sets    
train_array_x=train_set_x.astype('float32')
train_array_y=train_set_y.astype('float32') 
train_x_norm=train_array_x/255            


image_length=128 # length=width=128 
image_size_flat=49152 # 128*128
num_classes=62 # number of target classes
num_channels=3 #grayscale image        

# Creating functions for TensorFlow Model

def n_weights(shape): # Function to initialize random weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    
def n_biases(length): # Function to initialize Bias
    return tf.Variable(tf.constant(0.05, shape=[length]))    

# Function to initialize a new convolutional layer 
# parameters- input from previous layer, num_input_channels is the Num. channels in prev. layer, filter_size is the length/width of each filter/kernel, num_filters are the number of filters  
def new_conv_layer(input,num_input_channels, filter_size,num_filters,use_pooling=True): 

    # Shape of filter defined by the TensorFlow API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights for the filters with the given shape.
    weights = n_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = n_biases(length=num_filters)

    # Operation for ConvNet. Define Stride for the number of pixels moved over while performing convolution
    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1],padding='SAME')

    # Add the biases to the results of the convolution.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # 2x2 max-pooling is used which means max value is selected from the 2x2 grid to reduce the image size to significant features 
        layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

    layer = tf.nn.relu(layer)

    # We return both the resulting layer and the filter-weights
    return layer, weights 

# Function to flatten the output from the convolution layer as the next layer(Fully connected layer) require a 2-D shape
def flatten_layer(layer):
    
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

# Creating a fully connected layer (Normal layer in a neural network)    
def new_fc_layer(input,num_inputs,num_outputs,use_relu=True): 

    # Create new weights and biases.
    weights = n_weights(shape=[num_inputs, num_outputs])
    biases = n_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    # Relu
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer 
    
# Define Model parameters
    
# ConvNet 1.
filter_size1 = 8         # Convolution filters are 8 x 8 pixels.
num_filters1 = 16         # There are 16 of these filters.

# ConvNet 2.
filter_size2 = 5         # Convolution filters are 2 x 2 pixels.
num_filters2 = 36         # There are 36 of these filters.

# ConvNet 3.
filter_size3 = 2         # Convolution filters are 2 x 2 pixels.
num_filters3 = 64  

# Fully-connected layers.
fc_size1 = 256     
fc_size2=128   
    
# TensorFlow Graph

x = tf.placeholder(tf.float32, shape=[None, image_size_flat], name='X') # Placeholder for image data

x_img = tf.reshape(x, [-1, image_length, image_length, num_channels]) # Reshape image data into lenght X width X channels 

y = tf.placeholder(tf.float32 , shape=[None, num_classes], name='Y') # Placeholder for image target

y_true_cls = tf.argmax(y, axis=1) # Identify the class

# Creating layers in the network
conv_layer1, conv_weights1 = new_conv_layer(input=x_img,num_input_channels=num_channels,filter_size=filter_size1,num_filters=num_filters1,use_pooling=True)
                   
conv_layer2, conv_weights2 = new_conv_layer(input=conv_layer1,num_input_channels=num_filters1,filter_size=filter_size2,num_filters=num_filters2,use_pooling=True)

conv_layer3, conv_weights3 = new_conv_layer(input=conv_layer2,num_input_channels=num_filters2,filter_size=filter_size3,num_filters=num_filters3,use_pooling=True)

layer_flat, num_features = flatten_layer(conv_layer3)

layer_fc1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size1, use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,num_inputs=fc_size1,num_outputs=fc_size2, use_relu=True)

layer_fc3 = new_fc_layer(input=layer_fc2,num_inputs=fc_size2,num_outputs=num_classes, use_relu=False)

y_prob = tf.nn.softmax(layer_fc3,name='y_pred') # Apply softmax to the last layers' output

y_pred_cls = tf.argmax(y_prob, axis=1,name='predicted_class')

# Calculate the cross-entropy between true and predicted values
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,labels=y)

# Calculate the cross-entrpy error
cost = tf.reduce_mean(cross_entropy)

# Reduce the cost using Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start session to run the above TensorFlow graph
session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 30 # define bacth size
num_iterations=50 # Define number of iterations
Loss=[] # Initialize array to save loss after each iteration

# Initiate iterating
for i in range(num_iterations):
    j=0 # to change different batches
    while j<len(train_x_norm):
        start=j # Construct batches
        end=j+train_batch_size 
        batch_x,batch_y=train_x_norm[start:end],train_array_y[start:end] #  batches Constructed
        feed_dict_train = {x: batch_x, y: batch_y} # make dictionary to feed the placeholders in the model
        _,loss=session.run([optimizer,cost], feed_dict=feed_dict_train) # run optimizer and cost operations
        acc = session.run(accuracy, feed_dict=feed_dict_train) # run accuracy operation
        
        j+=train_batch_size # next batch
    Loss.append(loss)   # save loss    
    print('Loss at epoch',i,'is',loss,'and accuracy is',acc)

# Save the trained model    
saver = tf.train.Saver()
saver.save(session, 'char_recognition_CNN')     

# Saving train and test sets    
np.save('./train_x.npy',train_array_x)
np.save('./train_y.npy',train_array_y)
np.save('./test_x.npy',test_set_x)
np.save('./test_y.npy',test_set_y)

# Visualize the loss graph

plt.plot(Loss)
plt.show()
    

    
