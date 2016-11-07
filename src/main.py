import sys
import os.path
import random
import numpy as np
import timeit
from NN_BackPropagation import NN_BackPropagation
from NN_TruncatedNewton import NN_TruncatedNewton


def main(args):
    #Main funtion of the project. This is called when the program is started
    #from the command line.

    #Variable to calculate the duration of the run
    start = timeit.default_timer()

    fileName = args[1]
    fileMatrix = readFile(fileName, False)

    if fileMatrix is None or fileMatrix == []:
        print("""Input size columns does not correspond to the structure
 of the matrix""")
        return

    nnStructStr = args[2].\
        replace("[", "").replace("]", "").replace(" ", "").split(",")
    nnStructure = [int(x) for x in nnStructStr]

    if (len(fileMatrix[0]) - 1) != nnStructure[0]:
        print("""Input size columns does not correspond to the structure
 of the matrix""")
        return

    # Creates the training and testing sets. By default the NN is trained with
    # 80% of the input records and is tested with the remaining 20%
    trainingSet = []
    trainingExpectedSet = []

    testingSet = []
    testingExpectedSet = []

    for i in range(0, len(fileMatrix)):
        instance = fileMatrix[i]
        if i < len(fileMatrix) * 80 / 100:
            trainingSet.append(instance[:-1])
            trainingExpectedSet.append(instance[len(instance) - 1])
        else:
            testingSet.append(instance[:-1])
            testingExpectedSet.append(instance[len(instance) - 1])

    testType = args[3]
    if testType is None:
        testType = "ff"

    nn = None
    testTypeName = ""
    if testType.lower() == "tn":
        nn = NN_TruncatedNewton(nnStructure)
        testTypeName = "Truncated Newton Algorithm"
    elif testType.lower() == "bp":
        nn = NN_BackPropagation(nnStructure)
        trainingSet = np.array(trainingSet)
        trainingExpectedSet = np.array(trainingExpectedSet)
        testTypeName = "Back Propagation Algorithm"
    else:
        print("""Test type not valid. Please enter 'ff' or 'bp'""")
        return

    # Prints the input data info
    print(">> Input Information")
    print(("    Neural Network Configuration: " + str(nnStructure)))
    print(("    Solving Algorithm Chosen:     " + testTypeName))
    print(("    Total number of records:      " + str(len(fileMatrix))))
    print(("    Size of the training set:     " + str(len(trainingSet))))
    print(("    Size of the testing set:      " + str(len(testingSet))))
    print("")

    # Train the NN with the training set
    nn.train(trainingSet, trainingExpectedSet)

    #Now test the NN with the testing set
    output = nn.predict(testingSet, True)
    results = []
    numCorrect = 0.0
    for i in range(0, len(testingSet)):
        actual = output[i]
        expected = testingExpectedSet[i]
        correct = (actual + 0.0 == expected + 0.0)
        if correct is True:
            numCorrect += 1
        results.append(correct)

    #Accuracy #correct/total (in percentage)
    accuracy = (numCorrect / len(results)) * 100
    #Binomial Standard Deviation std=(P(1-P)/N)^0.5
    p = (len(results) - numCorrect) / len(results)
    std = (p * (1 - p) / len(results)) ** 0.5
    #Binomial variance var=np(1-p)
    var = len(results) * p * (1 - p)

    #Finally print out the results

    # Prints
    nn.printWeights()

    print(">> Results of training set:")
    print(("    " + (str(numCorrect) + " correct prediction(s) of " +
        str(len(testingSet)) + " testing records")))
    print(("    Accuracy:           {0:.2f}%".format(accuracy)))
    print(("    Standard Deviation: {0:.3f}".format(std)))
    print(("    Variance:           {0:.3f}".format(var)))
    print("")
    #Assigns the end time to calculate the total duration
    stop = timeit.default_timer()

    print(">> Total Duration:")
    print (("    {0:.3f}".format(stop - start) + " seconds"))
    print("")

    return


def readFile(fileName, shuffle):
    #Reads the file expecting n+1 columns where n is the size of the first
    #layer of the NN and the last column the label column (results)

    inputMatrix = []
    if(os.path.exists(fileName)):
        f = open(fileName, 'r')
        for line in f:
            line = line.replace("   ", " ")
            #print(line)
            lineNum = [float(x) for x in list(line.split())]
            inputMatrix.append(lineNum)

        if(shuffle):
            random.shuffle(inputMatrix)

        #Returns a simple array of arrays with all the data converted to doubles
        return inputMatrix

    else:
        print (("File " + fileName + " does not exist."))
        return None


if __name__ == '__main__':

    ##For testing
    #args_test = ["PATH", """TestData1024.dat", "[4,2,1]", "bp"]
    #main(args_test)

    main(sys.argv)
