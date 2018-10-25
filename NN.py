import numpy as np
import random

#Sigmoid Function
def sigmoid(x):
    return 1.0/ (1+np.exp(-x))

#Derivative Of The Sigmoid Function
def Dsigmoid(x):
    return x*(1.0-x)

class NeuralNetwork:

#The initialization of the Neural Network
    def __init__(self, input_size, num_hidden, output_size, num_epochs=1500, Learning_rate = 0.1):
        #Setting up the Neural Network
        self.final_output=0
        self.LR = Learning_rate
        self.num_inputs = input_size
        self.num_output = output_size
        self.num_hidden = num_hidden
        self.num_epochs = num_epochs

        #Randomizing the weights
        self.weights_ih = 2*np.random.rand(self.num_inputs,self.num_hidden)-1
        self.weights_ho = 2*np.random.rand(self.num_hidden, self.num_output)-1
        self.output = np.zeros(self.num_output)

# A function to calculate the accuracy of the Neural Network
    def acc(self, inputs, targets):
        #Running the inputs through the Neural network
        result = []
        true = 0
        Hlayer_output = np.dot(inputs, self.weights_ih)
        Hlayer_output = sigmoid(Hlayer_output)

        output = np.dot(Hlayer_output, self.weights_ho)
        output = sigmoid(output)

        #Comparing the result and the targets To Calculate the score
        for element in output:
            for value in element:
                result.append(int(round(value)))

        for i, element in enumerate(targets):
            for value in element:
                if value == result[i]:
                    true += 1.0
        score = true/len(result)
        return score


#A function the test the neural network on a given set of inputs
    def test(self,inputs):
        #Running the inputs through the Neural network
        result = []
        Hlayer_output = np.dot(inputs, self.weights_ih)
        Hlayer_output = sigmoid(Hlayer_output)

        output = np.dot(Hlayer_output, self.weights_ho)
        output = sigmoid(output)

        #Rounding the outputs to the closest integer EX: output=0.979, then output = 1
        for element in output:
            for value in element:
                result.append(int(round(value)))
        return result


#Training the Neural Network Using Backpropagation
    def backpropagate(self, inputs, targets):
        for i in range(self.num_epochs):
            #feeding tht inputs forward trhough the Neural Network
            Hlayer_output = np.dot(inputs, self.weights_ih)
            Hlayer_output = sigmoid(Hlayer_output)


            output = np.dot(Hlayer_output, self.weights_ho)
            output = sigmoid(output)

            #Calculating the Error
            output_errors = self.LR*(targets - output)

            #Calculating the gradient of the output layer
            gradient_output = output_errors * Dsigmoid(output)
            #Calculating the deltas of the output layer weights
            weightsHO_del = np.dot(Hlayer_output.T, gradient_output)

            #calculating the hidden layer error
            hidden_error = np.dot(self.LR*gradient_output, self.weights_ho.T)

            #Calculating the gradient of the hidden layer
            gradient_hidden = hidden_error * Dsigmoid(Hlayer_output)
            #Calculating the deltas of the hidden layer
            weightIH__del = np.dot(inputs.T, gradient_hidden)

            #updating the weights from the inputs layer to the hidden layer
            self.weights_ih += weightIH__del
            self.weights_ho += weightsHO_del
