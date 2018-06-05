import random
from copy import deepcopy

import gym
import math

import numpy


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

    def normiliseOutput(self, data):
        normalisedData = []

        data = numpy.asarray(data)
        for d in data:
            #for j in d:
            normalisedData.append(d/100)
        return normalisedData

    def activate(self, nodeVal, weights):
        return numpy.dot(nodeVal, weights)

    def sigmoid(self, s):
        #return 1 / (1 + numpy.exp(-s))
        return s/(1+abs(s))

    def predict(self, inputN, SimulationName):
        inputLayerOutput = self.sigmoid(self.activate(inputN, self.inputLayer))

        hiddenLayerOutput = []
        for i in range(self.neuronsHidden - 1):
            hiddenLayerOutput = self.sigmoid(self.activate(inputLayerOutput, self.hiddenLayers[i]))
            inputLayerOutput = deepcopy(hiddenLayerOutput)

        output = self.sigmoid(self.activate(hiddenLayerOutput, self.outputLayer))

        if SimulationName != 'MountainCarContinuous-v0':
            if output < 0.33:
                output = 0
            elif 0.33 < output < 0.66:
                output = 1
            else:
                output = 2
        return output

    def run(self, SimulationName):
        env = gym.make(SimulationName)
        observation = env.reset()
        fit = 0
        for t in range(1000):
            env.render()
            action = self.predict(observation, SimulationName)

            observation, reward, done, info = env.step(action)
            fit += reward
            if done:
                self.fit = abs(fit)
                print("Fitness: ", fit)
                print("Episode finished after {} timesteps".format(t + 1), " with fitness ", fit)
                env.close()
                break
        self.fit = abs(fit)
        print("Fitness: ", fit)
        env.close()

    def fitness(self):
        return self.fit


class Population:

    def __init__(self, sizePopulation, neuronInput, neuronHidden, neuronOutput, nrHidden):
        self.sizePopulation = sizePopulation
        self.population = [Individual(neuronInput, neuronHidden, neuronOutput, nrHidden) for _ in range(self.sizePopulation)]
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
        for i in range(parent1.neuronsHidden-1):
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
        factor = 2 * random.uniform(-1, 1) * self.lastBest/(self.currentBest+0.001)
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

        #for i in range(self.sizePopulation):
        #self.population[i].run()

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

    def selection(self, children, candidatesIndexes, SimulationName):
        j = 0
        for i in children:
            self.population[candidatesIndexes[j]].run(SimulationName)
            self.population[candidatesIndexes[j+1]].run(SimulationName)
            self.population[candidatesIndexes[j+2]].run(SimulationName)
            i.run(SimulationName)
            if self.population[candidatesIndexes[j]].fit < i.fit:
                self.population[candidatesIndexes[j]] = i
            else:
                if self.population[candidatesIndexes[j+1]].fit < i.fit:
                    self.population[candidatesIndexes[j+1]] = i
                else:
                    if self.population[candidatesIndexes[j+2]].fit < i.fit:
                        self.population[candidatesIndexes[j+2]] = i
            j += 3

    def best(self, n):
        aux = sorted(self.population, key=lambda Individual: Individual.fit)
        aux.reverse()
        return aux[:n]


def iteration(pop, Iteration, Input, Hidden, Output, Layers, SimulationName):
    p = Population(pop, Input, Hidden, Output, Layers)
    for i in range(Iteration):
        donorVector, indexes = p.evolve()
        p.selection(donorVector, indexes, SimulationName)
        offspringError = p.evaluate()
        p.lastBest = p.currentBest
        print("Best fitness: ", p.best(1)[0].fit)
        p.currentBest = p.best(1)[0].fit
        print("LOG Global Error")
        print(offspringError)
        p.population = p.best(20)

#iteration(20, 100, 2, 2, 1, 2, 'MountainCarContinuous-v0')
