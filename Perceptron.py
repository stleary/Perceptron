import sys
import getopt
import random
import math

class Perceptron:
    '''
    A perceptron:
        - Is the lowest level unit of an artificial neural network. In this exmaple, the perceptron executes alone.
        - Has exactly one output.
        - Can have any number of inputs. In this case, it has 2 inputs,
            to solve a straight line equation in the form y = mx + b
            where x and y are the inputs
        - Has a weight for each input. The weights are initialized to random values and will be adjusted over
            time as the perceptron is trained.
        - Has an optional bias, which allows you to offset the result by some amount.
            In this example, the bias is the y-intercept of a linear equation.
            Once set, the bias does not change.
        - Has a learning rate, so that corrections are neither too large, nor too small.
            Once set, the learning rate does not change.
    Any perceptron should have methods for initialization, activate(), train(), and query()
    '''
    def __init__(self, inputs: int=2, learning_rate: float=.01, bias: float=0.0):
        '''
        Initializes a new perceptron.
        :param inputs: Number of inputs for this instance
        :param learning_rate: How much to change the weights when a prediction is incorrect
        :param bias: offset for the activation function
        :param activation_function: an externally provided function that calculates what to return from a train() or query() call
        '''
        self.bias = bias
        self.learning_rate = learning_rate
        # create a list of weights, 1 weight for each input
        self.weights = [0] * inputs
        # initialize the weights to small, nonzero, random values
        for i in range(len(self.weights)):
            self.weights[i] = random.random() * 0.99 + .01  # sample from [0.01,1)

    def activate(self, value: float) -> float:
        '''
        This is the activation function. It returns the predicted result. In a neural network, this
        would be a sigmoid or some similar differentiable function that returns a value in the range (0,1).
        In the case of a single purpose perceptron, it is sufficient to use a simple step function
        that returns 0 or 1, by comparing the incoming value+bias to 0.
        :param value: sum of the weighted inputs
        :return 0 if the sum is less than 0, otherwise returns 1
        '''

        return 0 if value + self.bias < 0 else 1

        # this is the sigmoid function, in case you want to try it out
        # return 0 if (1 / (1 + math.exp(-1 * (value+self.bias)))) < .5 else 1

    def train(self, values: [], target: int):
        '''
        The perceptron can be trained by providing training data and an expected result. If it guesses
        wrong, the weight for each input will be adjusted.
        :param values: a list of input values, one for each weight
        :param target: the expected result (0 or 1)
        :return: the query result
        '''
        result = self.query(values)
        if (target != result):
            # the result was incorrect. Modify the weights according to whether 0 or 1 was expected
            # Change the weights according to the value * learning rate.
            for i in range(len(self.weights)):
                self.weights[i] += ((target - result) * values[i] * self.learning_rate)
            self.bias += ((target - result) * self.learning_rate)
        return result

    def query(self, values):
        '''
        The perceptron can be tested by providing training data but no expected result.
        Each raw input is modified by the weight for that input, then the sum of the inputs
        is passed to the activation function. The activation function decides how to classify the value.
        :param values: a list of input values, one for each weight
        :return: the result of the activation function
        '''
        weighted_sum = 0
        for i in range(len(self.weights)):
            weighted_sum += (self.weights[i] * values[i])
        return self.activate(weighted_sum)

def usage():
    print('Usage: python Perceptron -a A -b B [-c C] [--learn L] [--lowrange R] [--highrange S] [--correct T]')
    print('This perceptron generates random X,Y pairs in the specified range and separates them according to a standard form straight line equation ax + by = c')
    print('Parameters: x,y coefficients are required. The constant, learn, lowrange, highrange, and correct values are optional')
    print('   a: coefficient of X')
    print('   b: coefficient of Y')
    print('   c: constant term. Default is 0')
    print('   Learn: This is the learning rate, or how much the weights will change if the perceptron ')
    print('      guesses incorrectly. It should be a small value. Default is .005')
    print('   Lowrange: This is the low end of the range of generated X and Y values. Default is -10')
    print('   Highrange: This is the high end of the range of generated X and Y values. Default is 10')
    print('   Correct: This is the maximum number of correct guesses in a row, to terminate the test. Default is 100')
    print('Example: python Perceptron.py -a 1.4 -b -5 -c 13 --learn .0005 --lowrange -10 --highrange 10 --correct 200')
    print('   In this example, the perceptron solves for the inequality 1.4x -5y - 13 > 0')
    print('   Corrections to the weights are multiples of .0005')
    print('   The range of randomly generated input data is [-10,10]')
    print('   The test will stop when the trained perceptron has returned the correct result 200 times in a row')
    print('   The perceptron will learn to classify random X,Y values according to whether they satisfy the inequality, 1.4x -5y - 13 > 0')
    exit(2)


def main():
    '''
    Collect command line parameters, create Perceptron, train it until it can separate randomly generated x,y values
    100 times in a row. Show the actual and learned formulas in y = mx + b format. If -v is used, let the user enter
    some test x,y values.
    '''
    x_coefficient = None
    y_coeffient = None
    constant_term = 0
    learn = .005
    lowrange = -10
    highrange = 10
    correct = 100
    try:
        options, remainder = getopt.getopt(args=sys.argv[1:], shortopts='a:b:c:', longopts=['learn=', 'lowrange=', 'highrange=', 'correct='])
        for opt, arg in options:
            if opt in ('-a'):
                x_coefficient = float(arg)
            elif opt in ('-b'):
                y_coeffient = float(arg)
            elif opt in ('-c'):
                constant_term = float(arg)
            elif opt in ('--learn'):
                learn = float(arg)
            elif opt in ('--lowrange'):
                lowrange = float(arg)
            elif opt in ('--highrange'):
                highrange = float(arg)
            elif opt in ('--correct'):
                correct = float(arg)
        if remainder:
            usage()
        if not x_coefficient or not y_coeffient:
            usage()
        if not highrange > lowrange:
            usage()
    except Exception as e:
        usage()

    # The perceptron only needs to know the y-intercept, which is the bias from the origin
    y_intercept = constant_term / y_coeffient
    slope = -1 * x_coefficient / y_coeffient
    perceptron = Perceptron(inputs=2, bias=.1, learning_rate=learn)

    # x,y will be generated from random values in the range [lowrange, highrange] and used to train the perceptron
    i = 0
    count = 0
    while (count < correct):
        i += 1
        x = random.random() * (highrange - lowrange - .01) + .01 + lowrange
        y = random.random() * (highrange - lowrange - .01) + .01 + lowrange
        # we have to know the intended result in order to perform the training
        target = 1 if (x_coefficient * x + y_coeffient * y - constant_term > 0) else 0
        result = perceptron.train([x, y], target)
        print('{}. {} target: {} result: {} x: {} y: {} xwt {}  ywt {} bias {} {}'.
              format(i, str(result == target),
                     target,
                     result,
                     round(x, 2),
                     round(y, 2),
                     round(perceptron.weights[0], 2),
                     round(perceptron.weights[1], 2),
                     round(perceptron.bias, 2),
                     '**********************************************************' if result != target else ''))
        if result == target:
            count += 1
        else:
            count = 0
    print ("Predicted result correctly {} times in a row, after {} attempts".format(count, i))
    calculated_y_intercept = -perceptron.bias / perceptron.weights[1]
    calculated_slope = -perceptron.weights[0] / perceptron.weights[1]
    print('Actual  slope/intercept form : y = {}x {} {}'.format(round(slope,1),
                                                                    '+' if y_intercept > 0 else '-',
                                                                    round(abs(y_intercept),1)))
    print('Learned slope/intercept form : y = {}x {} {}'.format(round(calculated_slope,1),
                                                                    '+' if calculated_y_intercept > 0 else '-',
                                                                    round(abs(calculated_y_intercept),1)))




main()
