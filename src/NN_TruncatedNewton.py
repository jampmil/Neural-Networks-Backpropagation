from ffnet import ffnet, mlgraph


# Class that represents a Neural Network structure and able to handle a
# Truncated Newton algorithm to optimize the weights of the NN according
# an specific training set.
class NN_TruncatedNewton:

    def __init__(self, nnStructure):
        self.nnStructure = nnStructure
        conec = mlgraph(nnStructure)
        self.network = ffnet(conec)

    def train(self, nnInput, nnExpected):
        # Trains the network according the input set
        # In order to find the best starting point a genetic algorithm is used.
        self.network.train_genetic(nnInput, nnExpected)
        # train the network using the scipy tnc optimizer
        self.network.train_tnc(nnInput, nnExpected, maxfun=5000)

    def predict(self, nnInput, threshold=False):
        # Calculate the results of an specific input after going through
        # the trained Neural Network
        finalOutput = []

        # the test method requires an output. In our case we are just going to
        # ignore and calculate the stats locally
        output, regression = self.network.test(
            nnInput,
            [0.1] * len(nnInput),  # this target doesn't matter.
            iprint=0)

        for out in output:
            if threshold is False:
                finalOutput.append(out[0])
            elif out < 0.5:
                finalOutput.append(0)
            else:
                finalOutput.append(1)
        return finalOutput

    def printWeights(self):
        # Prints the weights of the Neural Network
        print(">>Weights of Neural Network:")

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
                        "{0:.2f}".format(self.network.weights[currentIndex])
                print(line)
            previousLayerSize = layerSize
        print("")
