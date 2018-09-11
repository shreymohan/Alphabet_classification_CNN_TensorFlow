# Alphabet Classification using CNNs from TensorFlow API

## Dataset Used 

The "62992 synthesised characters from computer fonts" dataset from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ has been used which has 62992 alphabets and letters in 62 different folders(classes) 0-9,a-z and A-Z. The training set is included in the repo. Extract the contents in the current folder.

## Issues

* The model has 3 ConvNet layers and 3 Max pooling layers and 3 fully connected layers (last one for output) which makes it very computationally intensive.
* The training was done on 60 letters&Alphabets each from every class which makes it 3720 training examples.
* It took around 6 hours for training the model for 50 iterations and with a batch size of 30.
* The use of GPUs would surely speed up the performance and accuracy.

## Results

* The training loss plot graph is included in the repo.
* The accuracy on the testing set which included 2 letters&Alphabets each from every 62 classes was 72.19%.
