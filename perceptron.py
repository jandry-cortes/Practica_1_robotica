import math
import numpy

class Perceptron:
    input = 0
    w1 = 0
    w2 = 0
    b1 = 0
    b2 = 0
    rate = 0.25
    output1 = 0
    output2 = 0

    def __init__(self, input):

            self.input = numpy.array(input).reshape(-1, 1)
            self.w1 = numpy.random.uniform(0,0.5,(3, 2))
            self.w2 = numpy.random.uniform(0,0.5,(2, 3))
            self.b1 = numpy.random.uniform(0,0.5,(3, 1))
            self.b2 = numpy.random.uniform(0,0.5,(2, 1))
            self.rate = 0.25
            self.calculateZ1()
            self.calculateZ2()

    def calculateFinalOutput(self, input):
        self.input = numpy.array(input).reshape(-1, 1)
        self.calculateZ1()
        self.calculateZ2()
        print(numpy.round(self.output2.reshape(1,-1), 0))

    def calculateOutput(self):
        self.calculateZ1()
        self.calculateZ2()

    def calculateZ1(self):
        z1 = numpy.matmul(self.w1, self.input) + self.b1
        self.output1 = 1 / (1 + numpy.exp(-z1))

    def calculateZ2(self):
        z2 = numpy.matmul(self.w2, self.output1) + self.b2
        self.output2 = 1 / (1 + numpy.exp(-z2))

    def backPropagation(self, input, y):
        self.input = numpy.array(input).reshape(-1, 1)
        y = numpy.array(y).reshape(-1, 1)

        self.calculateOutput()

        delta2 = self.calculateDelta2(y)
        self.actualizeWeight2(delta2)
        delta1 = self.calculateDelta1(delta2)
        self.actualizeWeight1(delta1)



    def getY(self):
        if self.input[0] == 0:
            if self.input[1] == 1:
                y = [1, 0]
            else:
                y = [0,0]
        else:
            if self.input[1] == 1:
                y = [0, 1]
            else:
                y = [1,0]
        return numpy.array(y).reshape(-1, 1)

    def calculateDelta2(self, y):
        return self.output2 - y

    def actualizeWeight2(self, delta2):
        partial_w2 = numpy.matmul(delta2, self.output1.T)
        self.w2 -= self.rate*partial_w2
        self.b2 -= self.rate*delta2

    def calculateDelta1(self, delta2):
        return numpy.matmul(self.w2.T, delta2) * self.output1 * (1 - self.output1)

    def actualizeWeight1(self, delta1):
        partial_w1 = numpy.matmul(delta1, self.input.T)
        self.w1 -= self.rate*partial_w1
        self.b1 -= self.rate*delta1
