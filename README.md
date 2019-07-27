## Perceptron ##
A simple, single-purpose Perceptron that can solve a linear equation of the form, y = mx + b

## Requires Python 3.6 or greater ##

Usage: python Perceptron.py --slope M --intercept I [--learn L] [--lowrange R] [--highrange R]

This perceptron generates random X,Y pairs in the specified range and separates them according to a linear equation of the form y = mx + b
   where x,y are the randomly generated values, m is the slope, and b is the y-intercept
   
Parameters: Slope and intercept are required, learn, lowrange, and highrange are optional with default values
* Slope: in the equation y = mx + b, slope is the m term
* Intercept: in the equation y = mx + b, intercept is the b term
* Learn: This is the learning rate, or how much the weights will change if the perceptron  guesses incorrectly. It should be a small value. Default is .005
* Lowrange: This is the low end of the range of generated X and Y values. Default is 0
* Highrange: This is the high end of the range of generated X and Y values. Default is 10

Example: python Perceptron.py --slope 1.4 --intercept 5 --learn .005 --lowrange -100, --highrange=100
* In this example, the perceptron solves for the inequality 1.4x - y > 5
* Corrections to the weights are multiples of .005
* The range of randomly generated input data is [-100,100]
* The perceptron will learn to classify random X,Y values according to whether they satisfy the inequality, 1.4x - y > 5 

The program will terminate after it has guessed correctly 100 times in a row.

