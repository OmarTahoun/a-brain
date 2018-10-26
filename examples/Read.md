# Hand written digits recognition
### In this example i use the [mnist images dataset](http://yann.lecun.com/exdb/mnist/) of hand written digits 
to train the neural network.


## About the data set:
- the data set consists of 28x28 images of hand written digits from 0 - 9
- the data contains 60,000 sample for training and 10,000 sample for testing

## Data preprocessing:
- Inputs and outputs must be in the format [n-samples, n-features]
- Outputs need to be a list of the probabilities of each digit.

## The training:
- we train the nn on the training data image by image
- the learning rate is set to 0.2 
- we will train the newtwork for 2 epochs since the data set is quite big (no need for more training)
