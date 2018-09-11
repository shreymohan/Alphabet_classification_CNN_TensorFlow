# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:25:36 2018

@author: shrey
"""

import numpy as np
import tensorflow as tf

# Loading the previously saved test set
test_x=np.load('./test_x.npy')
test_y=np.load('./test_y.npy')

# Change datatype and normalize testing sets
test_array_x=test_x.astype('float32')
test_array_y=test_y.astype('float32') 
test_x_norm=test_array_x/255  

# Restore and load the previous trained model 
sess=tf.Session() 
saver = tf.train.import_meta_graph('./char_recognition_CNN.meta') # Reloading meta data from saved model
saver.restore(sess,tf.train.latest_checkpoint('./'))


graph = tf.get_default_graph()  # get graph
# Get the placeholders
x = graph.get_tensor_by_name("X:0") 
y_true = graph.get_tensor_by_name('Y:0')

# Construct dictionary to feed the model
feed_dict_test={x:test_x_norm,y_true:test_array_y}

predict= graph.get_tensor_by_name("y_pred:0") # Get the prediction operation from the saved graph
predicted_class=graph.get_tensor_by_name("predicted_class:0") # Get the prediction operation from the saved graph

# Run the above two operations to get probabilities and predicted classes for the test set
y_pred,y_cls=sess.run([predict,predicted_class],feed_dict=feed_dict_test) # y_cls has the predicted classes on the test set

y_val_true=test_y==1 # get the position of 1s from the test set 
yind, y_true_class = np.where(test_y == 1) # get the true classes values from test set 

error=y_true_class-y_cls #  Subtract predicted from true class array
correct_pred=(error==0).sum() # Calculate number of zeroes - right predictions
test_accuracy=(float(correct_pred)/len(test_x))*100 # Calculate test accuracy of our model. 74.19% in this case

print(y_true_class) # True classes
print(y_cls) # predicted classes
print('Accuracy on test set is :'+str(test_accuracy)+'%') # accuracy of the model in predicting the right letters/alphabets out of all testing examples