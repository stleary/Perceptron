import sys
import getopt
import random
from typing import Callable

class Perceptron:
    '''
    A perceptron:
        - Is the lowest level unit of an artificial neural network. In this exmaple, the perceptron executes alone.
        - Has exactly one output.
        - Can have any number of inputs. In this case, it has 2 inputs,
            to solve a linear equation in the form y = mx + b
        - Has a weight for each input. The weight can be adjusted over time.
        - Has an optional bias, which allows you to offset the result by some amount.
            In this example, the bias is the y-intercept of a linear equation.
            The bias does not change.
        - Has a learning rate, so that corrections are neither too large, nor too small.
            The learning rate does not change.
    '''
    def __init__(self, inputs: int=2, learning_rate: float=.01, bias: float=0.0, activation_function: Callable = None):
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
        # initializat the weights to small, nonzero, random values
        for i in range(len(self.weights)):
            self.weights[i] = random.random() * 0.99 + .01  # sample from [0.01,1)

        self.activate = activation_function

    def activate(self, value: float) -> float:
        '''
        The activation function will be provided when the perceptron is created. That gives us
        flexibility to use different activation functions for different problems.
        :param value: the value to be tested by the activation function
        :return: about 0 for one of the binary classification choices, or about 1 for the other choice.
        '''
        return None

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
        return result

    def query(self, values):
        '''
        The perceptron can be tested by providing training data but no expected result.
        Each raw input is modified by the weight for that input, then the sum of the inputs
        is passed to the activation function.
        :param values: a list of input values, one for each weight
        :return: the result of the activation function
        '''
        weighted_sum = 0
        for i in range(len(self.weights)):
            weighted_sum += (self.weights[i] * values[i])
        return self.activate(weighted_sum)

def usage():
    print('Usage: python Perceptron --slope M --intercept I [--learn L] [--lowrange R] [--highrange R]')
    print('This perceptron generates random X,Y pairs in the specified range and separates them according to a linear equation of the form y = mx + b')
    print('   where x,y are the randomly generated values, m is the slope, and b is the y-intercept')
    print('Parameters: Slope and intercept are required, learn, lowrange, and highrange are optional with default values')
    print('   Slope: in the equation y = mx + b, slope is the m term')
    print('   Intercept: in the equation y = mx + b, intercept is the b term')
    print('   Learn: This is the learning rate, or how much the weights will change if the perceptron ')
    print('      guesses incorrectly. It should be a small value. Default is .005')
    print('   Lowrange: This is the low end of the range of generated X and Y values. Default is 0')
    print('   Highrange: This is the high end of the range of generated X and Y values. Default is 10')
    print('Example: python Perceptron.py --slope 1.4 --intercept 5 --learn .005 --lowrange -100, --highrange=100')
    print('   In this example, the perceptron solves for the inequality 1.4x - y > 5')
    print('   Corrections to the weights are multiples of .005')
    print('   The range of randomly generated input data is [-100,100]')
    print('   The perceptron will learn to classify random X,Y values according to whether they satisfy the inequality, 1.4x - y > 5 ')
    print('The program will terminate after it has guessed correctly 100 times in a row.')
    exit(2)


def main():
    slope = None
    intercept = None
    learn = .005
    lowrange = 0
    highrange = 10
    try:
        options, remainder = getopt.getopt(args=sys.argv[1:], shortopts='', longopts=['slope=', 'intercept=', 'learn=', 'lowrange=', 'highrange='])
        for opt, arg in options:
            if opt in ('--slope'):
                slope = float(arg)
            elif opt in ('--intercept'):
                intercept = float(arg)
            elif opt in ('--learn'):
                learn = float(arg)
            elif opt in ('--lowrange'):
                lowrange = float(arg)
            elif opt in ('--highrange'):
                highrange = float(arg)
        if remainder:
            usage()
        if not slope or not intercept:
            usage()
        if not highrange > lowrange:
            usage()
    except Exception as e:
        usage()

    """
        We will provide the perceptron's activation function.
        The acivation function returns the predicted result. In a neural network, this would be a sigmoid or
        some similar differentiable function. In the case of a single purpose perceptron, it is sufficient
        to use a simple step function that returns 0 or 1.
    """
    activation_function = lambda x: 0 if (x < intercept) else 1

    perceptron = Perceptron(inputs = 2, bias=intercept, learning_rate=learn, activation_function=activation_function)
    # f(x) = mx + b, where m is slope and b is intercept
    # x, y values will come from the range [lowrange, highrange]
    # errors will result in weights being adjusted by learn
    i = 0
    count = 0
    while (count < 100):
        i += 1
        x = random.random() * (highrange - lowrange - .01) + .01 + lowrange
        y = random.random() * (highrange - lowrange - .01) + .01 + lowrange
        target = 0
        if y > (slope * x + intercept):
            target = 1
        result = perceptron.train([x, y], target)
        print('{}. {} target: {} result: {} x: {} y: {} xwt {}  ywt {} {}'.
              format(i, str(result == target),
                     target,
                     result,
                     round(x, 2),
                     round(y, 2),
                     round(perceptron.weights[0], 2),
                     round(perceptron.weights[1], 2),
                     '**********************************************************' if result != target else ''))
        if result == target:
            count += 1
        else:
            count = 0
    print ("Predicted result correctly {} times in a row, after {} attempts".format(count, i))
main()
