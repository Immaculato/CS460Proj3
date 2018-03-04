#Tristan Basil
#Assignment: Project 3 - cS460G Machine Learning, Dr. Harrison
#https://stackoverflow.com/questions/3282823/get-the-key-corresponding-to-the-minimum-value-within-a-dictionary -
#used to find corresponding key to min value in a dictionary
#https://stackoverflow.com/questions/26584003/output-to-the-same-line-overwriting-previous
#used to print progress

import matplotlib.pyplot as plt
import numpy as np
import sys
import copy

#this class is only designed to work for the data in this project.
class PolynomialRegression:
    debug = False
    degree = 0
    alpha = 0.0
    cost = 0.0
    figureIndex = 0
    iterationCount = 0
    #list containing tuples for data
    ratings = list()
    #list containing weights for data
    weights = list()

    #initialization takes a filename.
    def __init__(self, degree, alpha, fileContents, debug):
        self.degree = degree
        self.alpha = alpha
        self.debug = debug
        #get all the ratings for each user.
        for line in fileContents:
            #print(line.rstrip())
            parsedLine = line.rstrip().split(',')
            #cast to float
            parsedLine[0] = float(parsedLine[0])
            parsedLine[1] = float(parsedLine[1])
            #mark the distinct movie indexes and user indexes.
            self.ratings.append(parsedLine)

        #initialize weights to 0.5 arbitrarily
        for i in range(self.degree+1):
            self.weights.append(0.0)

        #print(self.__hypothesis(0.5))
        self.cost = self.__cost()
        isGradientDescentFinished = False
        while not isGradientDescentFinished:
            self.iterationCount+=1
            isGradientDescentFinished = self.__gradientDescent()
            #if self.iterationCount % 1000 == 0:
            #    print self.iterationCount
            if self.iterationCount > 100000:
                isGradientDescentFinished = True

    def __hypothesis(self, x):
        hypothesisTot = 0.0
        for i in range(self.degree+1):
            hypothesisTot+=self.weights[i]*(x**i)
        return hypothesisTot

    def __gradientDescent(self):   
        newWeights = list()
        numRatings = len(self.ratings)
        jSum = 0.0
        numSmallChanges = 0
        for j in range(self.degree+1):
            for i in range(numRatings): 
                jSum += (1.0/numRatings) * (self.__hypothesis(self.ratings[i][0]) - self.ratings[i][1]) * (self.ratings[i][0]**j)

            #if the parameter change is small, note that it's small. 
            #print 'jSum', jSum
            if abs(jSum) < 0.01:
                numSmallChanges+=1
            
            newWeights.append(self.weights[j] - self.alpha*jSum)

        if self.debug:
            #print 'prev weights:', self.weights
            print 'new weights:', newWeights
        oldWeights = self.weights
        self.weights = newWeights

        #if all of our changes were very small, we're done.
        #print numSmallChanges
        #variable alpha
        newCost = self.__cost()
        #print newCost
        
        if newCost > self.cost:
            self.alpha = self.alpha * 0.5
            print 'newalpha', self.alpha
            self.weights=oldWeights
            return False
        
        #self.cost = newCost
        

        if numSmallChanges == self.degree+1:
            return True
        else:
            return False

    def __cost(self):
        numRatings = len(self.ratings)
        cost = 0.0
        for i in range(numRatings):
            cost+= (1.0/(2*numRatings)) * ((self.__hypothesis(self.ratings[i][0]) - self.ratings[i][1])**2)
        return cost

    def meanSquaredError(self):
        meanSquaredError = 0.0
        index = 0
        iteration = 0
        numRatings = len(self.ratings)
        for i in range(numRatings):
            prediction = self.__hypothesis(self.ratings[i][0])
            meanSquaredError += ((prediction - self.ratings[i][1]) ** 2)
        #print 'a', meanSquaredError, numRatings
        meanSquaredError = meanSquaredError/numRatings
        return meanSquaredError

    def printChart(self, chartName, degree):
        #plt.figure(figureIndex)

        #split out the features from the tuples into lists
        valueLists = list()
        for i in range(2):
            valueLists.append(list())
            numRatings = len(self.ratings)
            for j in range(numRatings):
                valueLists[i].append(self.ratings[j][i])

        plt.scatter(valueLists[0], valueLists[1], zorder=15)
        plt.hold(True)

        t1 = np.arange(min(valueLists[0]), max(valueLists[0]), 0.02)
        t2 = list()
        for i in range(len(t1)):
            t2.append(self.__hypothesis(t1[i]))
        #print t1
        #print t2
        plt.plot(t1, t2, 'k')
        plt.title('Dataset: '+chartName+', Degree = '+str(degree))
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.show()

    def getWeights(self):
        return self.weights


def main():
    if (len(sys.argv) != 2):
        print "Takes one command line argument: the name of the training file, the test file, and optionally, the -cv flag to cross validate."
        exit(-1)
    trainingFilename = sys.argv[1]

    #try to open the training file, and populate the array of lines.
    fileContents = list()
    try:
        fileTraining = open(trainingFilename, "r")
        for line in fileTraining:
            fileContents.append(line)
    except:
        print('training file not found')
        exit -1
    #7.8125e-05
    #0.01
    degrees = {9}
    for degree in degrees:
        regressionObject = PolynomialRegression(degree, 1, fileContents, debug=False)
        print 'Mean squared error for', trainingFilename, 'Degree', degree, regressionObject.meanSquaredError()
        print 'Weights', regressionObject.getWeights()
        regressionObject.printChart(trainingFilename, degree)

main()