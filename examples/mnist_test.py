from mnist import MNIST
from NN import *

#loading mnist data
def get_data():
    mndata = MNIST('./data')
    images_train, labels_train = mndata.load_training()
    images_test, label_test = mndata.load_testing()
    return images_train, labels_train, images_test, label_test
#Normalizing the Training and Testing data
x_train, y_train, x_test, y_test = get_data()
for i in range(len(x_train)):
    x_train[i] = [x / 255.0 for x in x_train[i]]
for i in range(len(x_test)):
    x_test[i] = [x / 255.0 for x in x_test[i]]


# Creating the neural network
nn = NeuralNetwork(784,64,10, num_epochs=4, Learning_rate = 2)

#Trining the network on the inputs and outputs
for j in range(nn.num_epochs):
    print("\nepoch: "+ str(j+1)+"\n")

    #Going through all the photos
    for i, photo in enumerate(x_train):
        inputs  = np.reshape(photo, (1,784))
        targets = [0,0,0,0,0,0,0,0,0,0]
        label = y_train[i]
        targets[label] = 1
        targets = np.reshape(targets, (1,10))
        #Training using Backpropagation
        nn.backpropagate(inputs, targets)

        #Printing out the progress of Training
        if i % 10000 == 0:
            print(str(i/600) + " % DONE")


#calculating the accuracy of the neural network using the test inputs and test labels
score  = nn.acc(x_test, y_test)
print(score)



#saving the module to json file for loading later
nn.save("nn-file.json")
