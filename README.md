## Perceptron ##
A simple, single-purpose Perceptron that can solve a straight line equation of the form, ax + by = c

## Requires Python 3.7 or greater ##

Usage: python Perceptron -a A -b B [-c C] [--learn L] [--lowrange R] [--highrange S] [--correct T]

This perceptron generates random X,Y pairs in the specified range and separates them according to a standard form straight line equation ax + by = c

Parameters: x,y coefficients are required. The constant, learn, lowrange, highrange, and correct values are optional
* a: coefficient of X
* b: coefficient of Y
* c: constant term. Default is 0
* Learn: This is the learning rate, or how much the weights will change if the perceptron guesses incorrectly. It should be a small value. Default is .005
* Lowrange: This is the low end of the range of generated X and Y values. Default is -10
* Highrange: This is the high end of the range of generated X and Y values. Default is 10
* Correct: This is the maximum number of correct guesses in a row, to terminate the test. Default is 100

Example: python Perceptron.py -a 1.4 -b -5 -c 13 --learn .0005 --lowrange -10 --highrange 10 --correct 200
* In this example, the perceptron solves for the inequality 1.4x -5y - 13 > 0
* Corrections to the weights are multiples of .0005
* The range of randomly generated input data is [-10,10]
* The test will stop when the trained perceptron has returned the correct result 200 times in a row
* The perceptron will learn to classify random X,Y values according to whether they satisfy the inequality, 1.4x -5y - 13 > 0
