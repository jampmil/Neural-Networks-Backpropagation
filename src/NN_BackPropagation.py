import numpy as np


# Class that represents a Neural Network structure and able to handle a
# back propagation algorithm to optimize the weights of the NN according
# an specific training set.
class NN_BackPropagation:

    def __init__(self, nnStructure, activation='Sigmoid', repetitions=100000):
        self.actFunction = activation
        self.repetitions = repetitions
        self.nnStructure = nnStructure

        # create the array of arrays for the weights
        self.weights = []

        # init the weights according the structure received by the input
        for i in range(0, len(nnStructure) - 1):
            # Creates random weights between 0 and 1
            layer = 2 * np.random.random((nnStructure[i],
                nnStructure[i + 1])) - 1
            self.weights.append(layer)

    def train(self, trainingSet, trainingExpectedSet, learning_rate=0.2):

        # start the repetitions
        for k in range(0, self.repetitions):

            # Randomize the training set for the current repetition
            rand = np.random.randint(trainingSet.shape[0])

            rowRes = [trainingSet[rand]]
            for layerIndex in range(len(self.weights)):
                    dot_value = np.dot(
                        rowRes[layerIndex],
                        self.weights[layerIndex])
                    act = self.activation(dot_value)
                    rowRes.append(act)

            # Calculates the error at the very end of the NN
            error = trainingExpectedSet[rand] - rowRes[-1]
            deltasFix = [error * self.activation(rowRes[-1], derivative=True)]

            # Now it calculates the error to be applied for each one of
            # the weights
            for l in range(len(rowRes) - 2, 0, -1):
                deltasFix.append(deltasFix[-1].dot(self.weights[l].T)
                    * self.activation(rowRes[l], derivative=True))

            # Now it runs back propagation
            # For this runs in reverse through the NN structure, calculating
            # the gradient (delta) and appliying it to all the weights
            # considering the learning factor
            deltasFix.reverse()
            for layerIndex in range(len(self.weights)):
                layer = np.atleast_2d(rowRes[layerIndex])
                delta = np.atleast_2d(deltasFix[layerIndex])
                self.weights[layerIndex] += learning_rate * layer.T.dot(delta)

    def predict(self, nnInput, threshold=False):
        # Calculate the results of an specific input after going through
        # the trained Neural Network
        output = []
        for nnIn in nnInput:
            a = nnIn
            for l in range(0, len(self.weights)):
                a = self.activation(np.dot(a, self.weights[l]))
            if threshold is False:
                output.append(a)
            elif a < 0.5:
                output.append(0)
            else:
                output.append(1)
        return output

    def printWeights(self):
        # Prints the weights of the Neural Network
        print(">>Weights of Neural Network:")

        # Converts the weights to a continuous array so we can use the
        # same print function we used in Feed Forward
        newWeights = []
        for layerIndex in range(0, len(self.weights)):
            layer = self.weights[layerIndex]
            m = len(list(layer[0]))  # num rows
            n = len(layer)  # Num columns
            for i in range(m):
                for j in range(n):
                    newWeights.append(list(layer[j])[i])

        # This is the same function we used for feed forward
        previousLayerSize = 0
        for layerIndex in range(0, len(self.nnStructure) - 1):
            layerSize = self.nnStructure[layerIndex]
            nextLayerSize = self.nnStructure[layerIndex + 1]
            for node in range(0, nextLayerSize):
                line = ""
                for i in range(0, layerSize):
                    currentIndex = i +\
                        layerSize * node +\
                        previousLayerSize * layerSize * layerIndex
                    line += "    W(" + \
                        str(layerIndex) + \
                        str(i) + \
                        "." + \
                        str(layerIndex + 1) + \
                        str(node) + \
                        ")  = " + \
                        "{0:.2f}".format(newWeights[currentIndex])
                print(line)
            previousLayerSize = layerSize
        print("")

    def activation(self, value, derivative=False):
        # Activation function. By default it uses the sigmoid function but it
        # could also work with TanH
        if self.actFunction == "sigmoid":
            if derivative is True:
                return self.activation(value) * (1.0 - self.activation(value))
            return 1.0 / (1.0 + np.exp(-value))
        else:
            if derivative is True:
                return 1.0 - value ** 2
            return np.tanh(value)
