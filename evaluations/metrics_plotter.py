#Code written by Jason Whitmore
#WWU CSCI 491/2/3

#This script creates a plot of a trained model's performance with provided f1, f2, recall, precision scores, and loss


#v 3.2.1
from matplotlib import pyplot

#v 1.14.5
import numpy as np

#v 0.25.3
import pandas as pd

NUM_SECONDS = 60

PLOT_TITLE = "Model performance on variable amounts of player input data"

X_AXIS_LABEL = "Seconds of player input data from match start, S"
X_AXIS_MIN = 0
X_AXIS_MAX = NUM_SECONDS + 15
X_AXIS_STEPSIZE = 15



Y_AXIS_LABEL = "Performance"
Y_AXIS_MIN = 0.9
Y_AXIS_MAX = 1
Y_AXIS_STEPSIZE = 0.01





results = np.array([[15, 0.9768, 0.9803, 0.9711, 0.9827, 0.0970],
                    [30, 0.9894, 0.9902, 0.9881, 0.9908, 0.0616],
                    [45, 0.9790, 0.9793, 0.9786, 0.9795, 0.0971],
                    [60, 0.9495, 0.9497, 0.95,   0.9499, 0.188]])

x_axis = results[:, 0:1]


f1_scores = results[:,1:2]

f2_scores = results[:,2:3]

precision = results[:,3:4]

recall = results[:,4:5]

loss = results[:,5:]


pyplot.scatter(x_axis, f1_scores)

pyplot.scatter(x_axis, f2_scores)

pyplot.scatter(x_axis, precision)

pyplot.scatter(x_axis, recall)

#Connect the dots with colored lines:

pyplot.plot(x_axis, f1_scores)

pyplot.plot(x_axis, f2_scores)

pyplot.plot(x_axis, precision)

pyplot.plot(x_axis, recall)



#Insert text, legends

pyplot.title(PLOT_TITLE)

pyplot.xlabel(X_AXIS_LABEL)
pyplot.xlim(X_AXIS_MIN, X_AXIS_MAX)
pyplot.xticks(np.arange(X_AXIS_MIN, X_AXIS_MAX + .01, X_AXIS_STEPSIZE))

pyplot.ylabel(Y_AXIS_LABEL)
pyplot.ylim(Y_AXIS_MIN,Y_AXIS_MAX)
pyplot.yticks(np.arange(Y_AXIS_MIN, Y_AXIS_MAX + 0.00001, Y_AXIS_STEPSIZE))

pyplot.legend(["F1 score", "F2 score", "Precision", "Recall"])

pyplot.show()