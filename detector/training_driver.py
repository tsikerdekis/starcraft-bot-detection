import detector

#v 1.14.5
import numpy as np

#v 1.10.0
import tensorflow as tf

#v 2.2.4
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LeakyReLU, Input, LSTM

import os
import sys

#Path to the folder containing the csv files
CSV_FOLDER = os.curdir + "/csvs/"

#Creating the detector
d = detector.detector()

#K folds training


d.k_folds_training(CSV_FOLDER, folds=10)
#d.predict_realtime(CSV_FOLDER + "15k/bot/b_10_p1.csv")

#Tests

#d.test_overfit(CSV_FOLDER, num_samples_list=[3000, 6000, 9000, 12000, 15000], num_steps=3000, minibatch_size=64)
#d.test_datashuffle(CSV_FOLDER)
#d.test_unit_tests()
