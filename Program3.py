#Tristan Basil
#Assignment: Project 3 - cS460G Machine Learning, Dr. Harrison

import matplotlib.pyplot as plt
import numpy as np
import sys

#https://stackoverflow.com/questions/3777861/setting-y-axis-limit-in-matplotlib
#for settings axes on charts
#

#this class is only designed to work for the data in this project.
class PolynomialRegression:
    debug = False
    order = 0
    alpha = 0.0
    cost = 0.0
    figureIndex = 0
    iterationCount = 0
    #list containing tuples for data
    ratings = list()
    #list containing weights for data
    weights = list()

    def __init__(self, order, alpha, fileContents, debug):
        self.order = order
        self.alpha = alpha
        self.debug = debug
        #for each line in the training file, parse out the line.
        for line in fileContents:
            parsedLine = line.rstrip().split(',')
            parsedLine[0] = float(parsedLine[0])
            parsedLine[1] = float(parsedLine[1])
            self.ratings.append(parsedLine)

        #initialize weights to 0.5 arbitrarily
        for i in range(self.order+1):
            self.weights.append(0.0)

        #get the initial cost and start using gradient descent.
        self.cost = self.__cost()
        isGradientDescentFinished = False
        #while not finished with gradient descent,
        while not isGradientDescentFinished:
            #keep doing iterations of gradient descent.
            self.iterationCount+=1
            isGradientDescentFinished = self.__gradientDescent()
            if self.debug and self.iterationCount % 1000 == 0:
                print 'Cost at iteration', self.iterationCount, '-', self.__cost()
            #if we hit the iteration limit, just stop early. it would converge if we let it, though.
            if self.iterationCount > 100000:
                isGradientDescentFinished = True

    def __hypothesis(self, x):
        hypothesisTot = 0.0
        for i in range(self.order+1):
            hypothesisTot+=self.weights[i]*(x**i)
        return hypothesisTot

    def __gradientDescent(self):   
        newWeights = list()
        numRatings = len(self.ratings)
        jSum = 0.0
        numSmallChanges = 0
        #for every order and rating, sum up the cost derivative with respect to that weight.
        for j in range(self.order+1):
            for i in range(numRatings): 
                jSum += (1.0/numRatings) * (self.__hypothesis(self.ratings[i][0]) - self.ratings[i][1]) * (self.ratings[i][0]**j)
            #if the parameter change is small, note that it's small. 
            if abs(jSum) < 0.01:
                numSmallChanges+=1
            newWeights.append(self.weights[j] - self.alpha*jSum)

        #if the new cost is worse, then change alpha and reset this iteration.
        oldWeights = self.weights
        self.weights = newWeights
        newCost = self.__cost()
        if newCost > self.cost:
            self.alpha = self.alpha * 0.5
            if self.debug:
                print 'New alpha:', self.alpha
            self.weights=oldWeights
            return False
        
        #if all weight changes are small, we're done!
        if numSmallChanges == self.order+1:
            return True
        else:
            return False

    #in hindsight, this is almost the exact same as mean squared error but is implemented differently.
    def __cost(self):
        numRatings = len(self.ratings)
        cost = 0.0
        for i in range(numRatings):
            cost+= (1.0/(2*numRatings)) * ((self.__hypothesis(self.ratings[i][0]) - self.ratings[i][1])**2)
        return cost

    def meanSquaredError(self):
        meanSquaredError = 0.0
        numRatings = len(self.ratings)
        for i in range(numRatings):
            prediction = self.__hypothesis(self.ratings[i][0])
            meanSquaredError += ((prediction - self.ratings[i][1]) ** 2)
        meanSquaredError = meanSquaredError/numRatings
        return meanSquaredError

    def printChart(self, chartName, order):

        #split out the features from the tuples into lists
        valueLists = list()
        for i in range(2):
            valueLists.append(list())
            numRatings = len(self.ratings)
            for j in range(numRatings):
                valueLists[i].append(self.ratings[j][i])

        plt.scatter(valueLists[0], valueLists[1], zorder=15)
        plt.hold(True)

        xValues = np.arange(min(valueLists[0])-0.5, max(valueLists[0])+0.5, 0.02)
        yHypothesis = list()
        axes = plt.gca()
        axes.set_xlim(min(valueLists[0])-0.5, max(valueLists[0])+0.5)
        axes.set_ylim(min(valueLists[1])-0.5, max(valueLists[1])+0.5)
        for i in range(len(xValues)):
            yHypothesis.append(self.__hypothesis(xValues[i]))
        plt.plot(xValues, yHypothesis, 'k')
        plt.title('Dataset: '+chartName+', Order = '+str(order))
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.show()

    def getWeights(self):
        return self.weights


def main():
    if (len(sys.argv) != 3):
        print "Takes two command line arguments: the name of the training file, and the order."
        exit(-1)
    trainingFilename = sys.argv[1]
    order = int(sys.argv[2])

    #try to open the training file, and populate the array of lines.
    fileContents = list()
    try:
        fileTraining = open(trainingFilename, "r")
        for line in fileTraining:
            fileContents.append(line)
    except:
        print('training file not found')
        exit -1

    regressionObject = PolynomialRegression(order, 1, fileContents, debug=True)
    print 'Mean squared error for', trainingFilename, 'Order', order, regressionObject.meanSquaredError()
    print 'Weights', regressionObject.getWeights()
    regressionObject.printChart(trainingFilename, order)

main()