from mnist import MNIST
from NN import *

#loading mnist data
def get_data():
    mndata = MNIST('./data')
    images_train, labels_train = mndata.load_training()
    images_test, label_test = mndata.load_testing()
    return images_train, labels_train, images_test, label_test

x_train, y_train, x_test, y_test = get_data()

#setting the neural network
nn = NeuralNetwork(784,16,10, num_epochs=2, Learning_rate = 0.2)

#Trining the network o the inputs and outputs
for j in range(nn.num_epochs):
    print("epoch: "+ str(j)+"\n")

    #Going through all the photos
    for i, photo in enumerate(x_train):
        inputs  = np.reshape(photo, (1,784))
        targets = [0,0,0,0,0,0,0,0,0,0]
        label = y_train[i]
        targets[label] = 1
        targets = targets = np.reshape(targets, (1,10))
        #Training using Backpropagation
        nn.backpropagate(inputs, targets)

        #Printing out the progress of Training
        if i % 10000 == 0:
            print(str(i/600) + " % DONE")


#Testing with test data
result = nn.test(x_test[1200])
guess = np.argmax(result)
actual = y_test[1200]

#Printing the guess and the actual Label
print("You guessed : " + str(guess))
print("the actual result is: " + str(actual))
