import numpy
import math
import random
import pandas
from copy import deepcopy
from sklearn.model_selection import train_test_split


numpy.seterr(all='raise')


class Individual:

    def __init__(self, nInput, nHidden, nOutput, nrHidden):
        self.neuronsInput = nInput
        self.neuronsOutput = nOutput
        self.neuronsHidden = nHidden
        self.nrHidden = nrHidden

        self.fit = 0

        self.inputLayer = []
        self.hiddenLayers = []
        self.outputLayer = []

        self.inputLayer = 2 * numpy.random.random((self.neuronsInput, self.neuronsHidden)) - 1
        for i in range(self.nrHidden-1):
            weights = 2 * numpy.random.random((self.neuronsHidden, self.neuronsHidden)) - 1
            self.hiddenLayers.append(weights)

        self.outputLayer = 2 * numpy.random.random((self.neuronsHidden, self.neuronsOutput)) - 1

        self.inputLayer = numpy.asarray(self.inputLayer)
        self.outputLayer = numpy.asarray(self.outputLayer)

        self.virtualInput = [[i] for i in range(1, 1400)]
        self.virtualOutput = [math.sin(i) for i in range(1, 1400)]


        self.input = self.readCSV('./data/input.xlsx')
        self.output = self.normiliseOutput(self.readCSV('./data/output.xlsx'))
        self.test = self.readCSV('./data/actual_test_values.xlsx', end=2077)
        self.y_test = self.normiliseOutput(self.readCSV('./data/actual_test_output.xlsx', end=2077))

        #self.input, self.test, self.outputNodes, self.y_test = train_test_split(self.input, self.output, test_size=0.66, random_state=42)

        data = pandas.ExcelFile('./data/output.xlsx')
        sheet = data.parse('Sheet1')
        self.outputNodes = self.normiliseOutput(sheet.values.tolist()[:1400])  # 900-1400

        #print(len(self.input), " ", len(self.outputNodes))
        self.input = numpy.asarray(self.input, dtype=numpy.float64)

        self.inputNodes = []
        for i in self.input:
            l = i.flatten()
            l = numpy.append(1, l)
            self.inputNodes.append(l)

        
        self.testData = []
        self.test = numpy.asarray(self.test, dtype=numpy.float64)
        for i in self.test:
            l = i.flatten()
            l = numpy.append(1, l)
            self.testData.append(l)

        #print(len(self.testData))
        self.outputNode = numpy.asarray(self.outputNodes, dtype=numpy.float64)
        self.y_test = numpy.asarray(self.y_test, dtype=numpy.float64)

    def readCSV(self, filename, start=0, end=1400):
        return pandas.read_excel(filename)[start:end]
        #return pandas.read_excel(filename, header=None)

    def normiliseOutput(self, data):
        data = numpy.asarray(data)
        xMax = max(data)
        xMin = min(data)

        normalisedData = []

        data = numpy.asarray(data)
        
        for d in data:
            normalisedData.append((d-xMin)/(xMax - xMin))

        return normalisedData

    def normiliseOutputOld(self, data):
        normalisedData = []

        data = numpy.asarray(data)
        
        mean = 0
        size = 0
        deviation = []
        for d in data:
            #for j in d:
            mean += d
            size += 1
            deviation.append(d)
        
        mean = mean / size

        standardDeviation = 0
        for i in deviation:
            standardDeviation += (i - mean) ** 2

        standardDeviation = math.sqrt(standardDeviation/len(deviation))

        normalisedData = []
        for i in deviation:
        
            normalisedData.append((i - mean)/standardDeviation)

        #normalisedData.append(d/100)
        return normalisedData

    def activate(self, nodeVal, weights):
        return numpy.dot(nodeVal, weights)

    def tahn(self, s):
        return (numpy.exp(s) - numpy.exp(-s)) / (numpy.exp(s) + numpy.exp(-s))
        #s = [math.ceil((a*1e15)/1e15) for a in s]
        #s = numpy.asarray(s)
        #return 1 / (1 + numpy.exp(-s))

    def sigmoid(self, s):
        #return (numpy.exp(s) - numpy.exp(-s)) / (numpy.exp(s) + numpy.exp(-s))
        return 1 / (1 + numpy.exp(-s))

    def fitness(self, test=False):
        fit = 0
        candidate = 0
        #for inputN in self.virtualInput:
        if test == False:
            for inputN in self.inputNodes:
                try:
                    inputLayerOutput = self.sigmoid(self.activate(inputN, self.inputLayer))

                    hiddenLayerOutput = []
                    for i in range(self.nrHidden - 1):
                        hiddenLayerOutput = self.sigmoid(self.activate(inputLayerOutput, self.hiddenLayers[i]))
                        inputLayerOutput = deepcopy(hiddenLayerOutput)

                    output = self.sigmoid(self.activate(inputLayerOutput, self.outputLayer))
                    fit += (self.outputNode[candidate] - output) ** 2
                except FloatingPointError:
                    print("ENTERED ERROR")
                    fit += 10000
                candidate += 1
        else:
            for inputN in self.virtualInput:
                try:
                    inputLayerOutput = self.tahn(self.activate(inputN, self.inputLayer))

                    hiddenLayerOutput = []
                    for i in range(self.nrHidden - 1):
                        hiddenLayerOutput = self.tahn(self.activate(inputLayerOutput, self.hiddenLayers[i]))
                        inputLayerOutput = deepcopy(hiddenLayerOutput)

                    output = self.tahn(self.activate(inputLayerOutput, self.outputLayer))
                    fit += (int(self.virtualOutput[candidate]*10) - int(output*10)) ** 2
                except FloatingPointError:
                    #print("ENTERED ERROR")
                    fit += 100000
                candidate += 1

        self.fit = fit
        return self.fit

    def reMutate(self):
        index_value = random.sample(list(enumerate(self.inputLayer)), 2)
        for i in index_value:
            self.inputLayer[i[0]] = 2 * numpy.random.random() - 1

        index_value = random.sample(list(enumerate(self.outputLayer)), 2)
        for i in index_value:
            self.outputLayer[i[0]] = 2 * numpy.random.random() - 1

        for i in range(len(self.hiddenLayers)):
            index_value = random.sample(list(enumerate(self.hiddenLayers[i])), 2)
            for j in index_value:
                self.hiddenLayers[i][j[0]] = 2 * numpy.random.random() - 1

    def checkSolution(self, inputN, test=False):
        if test == False:
            try:
                inputLayerOutput = self.sigmoid(self.activate(inputN, self.inputLayer))

                #print("STARTED")
                #print(inputLayerOutput)

                hiddenLayerOutput = []
                for i in range(self.nrHidden - 1):
                    hiddenLayerOutput = self.sigmoid(self.activate(inputLayerOutput, self.hiddenLayers[i]))
                    inputLayerOutput = deepcopy(hiddenLayerOutput)

                #print(hiddenLayerOutput)

                output = self.sigmoid(self.activate(inputLayerOutput, self.outputLayer))
                #print("ENDED")
                
                return output
            except FloatingPointError:
                return -10
        else:
            try:
                inputLayerOutput = self.tahn(self.activate(inputN, self.inputLayer))

                #print("STARTED")
                #print(inputLayerOutput)

                hiddenLayerOutput = []
                for i in range(self.nrHidden - 1):
                    hiddenLayerOutput = self.tahn(self.activate(inputLayerOutput, self.hiddenLayers[i]))
                    inputLayerOutput = deepcopy(hiddenLayerOutput)

                #print(hiddenLayerOutput)

                output = self.tahn(self.activate(inputLayerOutput, self.outputLayer))
                #print("ENDED")
                
                return output
            except FloatingPointError:
                return -10

    def testAlgoritm(self, test=False):

        f = open("UItest.txt", 'w')
        f2 = open("UItestRealValues.txt", 'w')
        output = []
        output2 = []
        self.virtualInput = [[i] for i in range(1500, 2030)]
        self.virtualOutput = [math.sin(i) for i in range(1500, 2030)]
        if test == False:
            for i in range(len(self.testData)):
                output.append(str(self.checkSolution(self.testData[i])))
                output2.append(str(self.y_test[i]))
        else:
            for i in range(len(self.virtualInput)):
                output.append(str(self.checkSolution(self.virtualInput[i], test=True)))
                output2.append(str(self.virtualOutput[i]))
        f.write(str(output))
        f2.write(str(output2))

        return output


class Population:

    def __init__(self, sizePopulation, neuronInput, neuronHidden, neuronOutput, nrHidden):
        self.sizePopulation = sizePopulation
        self.population = [Individual(neuronInput, neuronHidden, neuronOutput, nrHidden) for i in range(self.sizePopulation)]
        self.lastBest = 1
        self.currentBest = 1

        self.nInput = neuronInput
        self.nOutput = neuronOutput
        self.nHidden = neuronHidden
        self.nrHidden = nrHidden

    def evaluate(self):
        sum = 0
        for x in self.population:
            sum += x.fit
        return sum

    def reMutatePopulation(self):
        for i in range(len(self.population)):
            if 0.5 > numpy.random.random():
                self.population[i].reMutate()

    def equationInput(self, parent1, parent2, candidate, Factor):
        list = []
        mutationProb = Factor/2
        for i in range(parent1.neuronsInput):
            l = []
            for j in range(parent1.neuronsHidden):
                prob = random.random()
                if prob > mutationProb:
                    nr = (parent2.inputLayer[i][j] - candidate.inputLayer[i][j]) * Factor + parent1.inputLayer[i][j]
                    l.append(nr)
                else:
                    l.append(candidate.inputLayer[i][j])
            list.append(l)
        return list

    def equationHidden(self, parent1, parent2, candidate, Factor):
        mutationProb = Factor/2
        mutatedLayers = []
        for i in range(parent1.nrHidden-1):
            l = []
            for j in range(parent1.neuronsHidden):
                prob = random.random()
                if prob > mutationProb:
                    nr = (parent2.hiddenLayers[i][j] - candidate.hiddenLayers[i][j]) * Factor + parent1.hiddenLayers[i][j]
                    l.append(nr)
                else:
                    l.append(candidate.hiddenLayers[i][j])
            mutatedLayers.append(numpy.asarray(l))
        return mutatedLayers

    def equationOutput(self, parent1, parent2, candidate, Factor):
        list = []
        mutationProb = Factor/2
        for i in range(parent1.neuronsHidden):
            l = []
            for j in range(parent1.neuronsOutput):
                prob = random.random()
                if prob > mutationProb:
                    nr = (parent2.outputLayer[i][j] - candidate.outputLayer[i][j]) * Factor + parent1.outputLayer[i][j]
                    l.append(nr)
                else:
                    l.append(candidate.outputLayer[i][j])
            list.append(l)
        return list

    def mutate(self, parent1, parent2, candidate):
        donorVector = Individual(self.nInput, self.nHidden, self.nOutput, self.nrHidden)
        factor = 2 * random.uniform(-1, 1) * self.lastBest/self.currentBest
        donorVector.inputLayer = numpy.asarray(self.equationInput(parent1, parent2, candidate, factor))
        donorVector.hiddenLayers = numpy.asarray(self.equationHidden(parent1, parent2, candidate, factor))
        donorVector.outputLayer = numpy.asarray(self.equationOutput(parent1, parent2, candidate, factor))

        return donorVector

    def crossover(self, individ1, donorVector):
        crossoverRate = 0.5

        trialVector = Individual(self.nInput, self.nHidden, self.nOutput, self.nrHidden)

        for i in range(len(individ1.inputLayer)):
            for j in range(len(individ1.inputLayer[i])):
                if random.random() > crossoverRate:
                    trialVector.inputLayer[i][j] = individ1.inputLayer[i][j]
                else:
                    trialVector.inputLayer[i][j] = donorVector.inputLayer[i][j]

        for i in range(len(individ1.hiddenLayers)):
            for j in range(len(individ1.hiddenLayers[i])):
                if random.random() > crossoverRate:
                    trialVector.hiddenLayers[i][j] = individ1.hiddenLayers[i][j]
                else:
                    trialVector.hiddenLayers[i][j] = donorVector.hiddenLayers[i][j]

        for i in range(len(individ1.outputLayer)):
            for j in range(len(individ1.outputLayer[i])):
                if random.random() > crossoverRate:
                    trialVector.outputLayer[i][j] = individ1.outputLayer[i][j]
                else:
                    trialVector.outputLayer[i][j] = donorVector.outputLayer[i][j]

        return trialVector

    def evolve(self):
        childred = []
        indexes = []
        for i in range(self.sizePopulation):
            #candidate = self.population[i]
            parents = random.sample(list(enumerate(self.population)), 3)
            parent1Index, parent1 = parents[0]
            parent2Index, parent2 = parents[1]
            candidateIndex, candidate = parents[2]

            while parent1 == candidate or parent2 == candidate or parent1 == parent2:
                parents = random.sample(self.population, 2)
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

            child = self.mutate(parent1, parent2, candidate)
            childCandidate = self.crossover(candidate, child)
            childred.append(childCandidate)

            indexes.append(parent1Index)
            indexes.append(parent2Index)
            indexes.append(candidateIndex)

        return childred, indexes

    def selection(self, children, candidatesIndexes, test=False):
        for i in children:
            self.population[candidatesIndexes[0]].fitness(test)
            self.population[candidatesIndexes[1]].fitness(test)
            self.population[candidatesIndexes[2]].fitness(test)
            i.fitness(test)
            if self.population[candidatesIndexes[0]].fit > i.fit:
                self.population[candidatesIndexes[0]] = i
            else:
                if self.population[candidatesIndexes[1]].fit > i.fit:
                    self.population[candidatesIndexes[1]] = i
                else:
                    if self.population[candidatesIndexes[2]].fit > i.fit:
                        self.population[candidatesIndexes[2]] = i

    def best(self, n, test=False):
        aux = sorted(self.population, key=lambda Individual: Individual.fitness(test))
        return aux[:n]


class Algorithm:

    def __init__(self, sizePop, generations, neuronInput, neuronHidden, neuronOutput, nrHidden):
        self.population = Population(sizePop, neuronInput, neuronHidden, neuronOutput, nrHidden)
        self.sizePop = sizePop
        self.generations = generations

    def iteration(self, test=False):

        donorVector, indexes = self.population.evolve()
        self.population.selection(donorVector, indexes, test)
        offspringError = self.population.evaluate()
        self.population.lastBest = self.population.currentBest
        print("Best Individual: ", self.population.best(1, test)[0].fit)
        self.population.currentBest = self.population.best(1, test)[0].fit
        print("LOG Global Error")
        print(offspringError/self.sizePop)

        #if self.population.currentBest >= offspringError/self.sizePop - 1:
        #    self.population.reMutatePopulation()

    def testRun(self, test=False):
        file = open("Final_Appended.txt", "a")
        file.write('\n')

        for k in range(self.generations):
            print("Iteration: ", k)

            self.iteration(test)
        return self.population.best(10, test)


#a = Algorithm(10, 20, 1, 2, 1, 2)

#a.testRun(test=True)

#a.population.population[0].testAlgoritm(test=True)
