#Code written by Jason Whitmore
#WWU CSCI 491/2/3

#This script creates a "real time" plot of predictions based on csv files that have columns
#[timestep, human mean, human stddev, bot mean, bot stddev]

#v 3.2.1
from matplotlib import pyplot

#v 1.14.5
import numpy as np

#v 0.25.3
import pandas as pd

SECONDS_OF_DATA = 60

REALTIME_PREDICTIONS_CSV_PATH = "realtime/realtime_" + str(SECONDS_OF_DATA) + ".csv"

FRAME_SCALAR = 1.0 / 24.0

HUMAN_COLOR = "green"

BOT_COLOR = "red"

ALPHA = 0.5



PLOT_TITLE = "Real time detector predictions (trained on " + str(SECONDS_OF_DATA) + " seconds of gameplay)"


X_AXIS_LABEL = "Time from start of match (s)"

X_AXIS_MIN = 0

X_AXIS_MAX = SECONDS_OF_DATA

X_AXIS_STEPSIZE = 5


Y_AXIS_LABEL = "Detector prediction"

Y_AXIS_MIN = 0

Y_AXIS_MAX = 1

Y_AXIS_STEPSIZE = 0.1


#Load into a numpy array

data = pd.read_csv(REALTIME_PREDICTIONS_CSV_PATH).values

#Scale the frame number column to be in terms of seconds passed

for i in range(len(data)):
    data[i][0] *= FRAME_SCALAR

#Isolate the rows and insert into own np arrays

seconds = []

h_mean = []
h_std = []

b_mean = []
b_std = []

for i in range(len(data)):
    seconds.append(data[i][0])

    h_mean.append(data[i][1])
    h_std.append(data[i][2])

    b_mean.append(data[i][3])
    b_std.append(data[i][4])

seconds = np.array(seconds)

h_mean = np.array(h_mean)
h_std = np.array(h_std)

b_mean = np.array(b_mean)
b_std = np.array(b_std)


pyplot.plot(seconds, h_mean, HUMAN_COLOR)
pyplot.fill_between(seconds, h_mean + h_std, h_mean - h_std, color=HUMAN_COLOR, alpha=ALPHA)

pyplot.plot(seconds, b_mean, BOT_COLOR)
pyplot.fill_between(seconds, b_mean + b_std, b_mean - b_std, color=BOT_COLOR, alpha=ALPHA)

pyplot.title(PLOT_TITLE)

pyplot.xlabel(X_AXIS_LABEL)
pyplot.xlim(X_AXIS_MIN, X_AXIS_MAX)
pyplot.xticks(np.arange(X_AXIS_MIN, X_AXIS_MAX + 1, X_AXIS_STEPSIZE))

pyplot.ylabel(Y_AXIS_LABEL)
pyplot.ylim(0,1)
pyplot.yticks(np.arange(Y_AXIS_MIN, Y_AXIS_MAX + 0.1, Y_AXIS_STEPSIZE))

pyplot.legend(["Human predictions", "Bot predictions"])

pyplot.show()
