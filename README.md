## Perceptron ##
A simple, single-purpose Perceptron that can solve a straight line equation of the form, ax + by = c

## Requires Python 3.7 or greater ##

Usage: python Perceptron -a A -b B [-c C] [--learn L] [--lowrange R] [--highrange S]

This perceptron generates random X,Y pairs in the specified range and separates them according to a standard form straight line equation ax + by = c

Parameters: x,y coefficients are required. The constant, learn, lowrange, and highrange values are optional
* a: coefficient of X
* b: coefficient of Y
* c: constant term. Default is 0
* Learn: This is the learning rate, or how much the weights will change if the perceptron guesses incorrectly. It should be a small value. Default is .005
* Lowrange: This is the low end of the range of generated X and Y values. Default is 0
* Highrange: This is the high end of the range of generated X and Y values. Default is 10

Example: python Perceptron.py -a 1.4 -b -5 -c 13 --learn .005 --lowrange -100 --highrange=100
* In this example, the perceptron solves for the inequality 1.4x -5y > 13
* Corrections to the weights are multiples of .005
* The range of randomly generated input data is [-100,100]
* The perceptron will learn to classify random X,Y values according to whether they satisfy the inequality, 1.4x -5y + 13 > 0

The program will terminate after it has guessed correctly 100 times in a row.
