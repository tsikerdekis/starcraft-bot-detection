"""
Joshua Orritt
Raleigh Hansen
Western Washington University
CSCI 493   Spring 2020

This program consists of 3 tasks:
	-1. Process data given by a numpy array:
		-beginning negative one's become 0.5
		-in gaps between "actual predictions", treat as if there was a linear line connecting them
		-after last actual prediction, insert that prediction over the remaining bad values
	-2. Group data and determine statistics
		-step one should be repeated for every game, storing all arrays in memory
		-group into games predicted to be human and predicted to be bot (>0.5 is bot)
		-get mean, standard deviation for each of 24 * SEQUENCE_LENGTH timestamps
	-3. Output to CSV
		-format: [game frame, human mean, human stddev, bot mean, bot stddev]
"""

import numpy as np
import statistics as st
import os
import csv

from detector import detector
Detector = detector()

#Path to the folder containing the csv files
HUMAN_FOLDER = os.curdir + "/../detector/csvs/human"
BOT_FOLDER = os.curdir + "/../detector/csvs/bot"
EVAL_PATH = os.curdir + "/realtime/sim_rt_eval.csv"

# drives evaluation process
#
# input: seq_len, number of seconds in a sequence
def simulate_realtime(seq_len):

	# WARNING: comment the following line out to preserve past evals
	clear_past_evals()

	h_mean, h_stddev = pred_metrics(HUMAN_FOLDER, seq_len)
	b_mean, b_stddev = pred_metrics(BOT_FOLDER, seq_len)
	populate_csv(h_mean, h_stddev, b_mean, b_stddev)


# computes mean and standard deviation at each timestep
# within replay sequences given a path to class of replays
#
# input: replay_path, string denoting path to class of replays
# output: mean, 1d numpy array of mean value over every timestep
# 	      stddev, 1d numpy array of standard deviation over every timestep
def pred_metrics(replay_path, seq_len):

	# gathers processed "real-time" predictions into numpy array
	# of dimension (num_replays, 24*seq_len)
	for root, directory, files in os.walk(replay_path):
		seqs = []
		for filename in files:
			print(filename)
			i = 0
			full_path = replay_path + "/" + filename
			realtime_array = Detector.predict_realtime(full_path)
			if realtime_array is not None:
				seqs.append(process_data(realtime_array))
			i+=1

	# compute mean and standard deviaton for each timestep
	# store in list

	seqs = np.array(seqs)
	mean = [None] * len(seqs[0,:])
	
	stddev = [None] * len(seqs[0,:])
	for col in range(len(seqs[0,:])):
		mean[col] = np.mean(seqs[:,col])
		stddev[col] = np.std(seqs[:,col])
	
	return mean, stddev


# alter bad values given by predictions to proper non-negative value
# 
# input: arr, numpy array to be processed
# output: arr, processed array
def process_data(arr):
	I = 0
	end_of_arr = arr.size
	end_of_data = end_of_arr

	# get index of last good value
	for end in range(end_of_arr - 1, 0, -1):
		if arr[end] != -1:
			break

	#----case 1: handle leading values
	while arr[I] == -1:
		arr[I] = 0.5
		I = I + 1
	
	index = I
	#----case 2: handle gap values
	for i in range(I,end):
		if arr[i+1] == -1:
			#find next good value
			j = 1
			while arr[i+j] == -1:
				j+=1
			next_value = arr[i+j]
			#get the numbers inbetween (this will include the 2 valid numbers we had)
			fillers = np.linspace(arr[i], arr[i+j], num= j + 1)
			#fill everything in
			l = 0
			for k in range(i, i+j):
				arr[k] = fillers[l]
				l = l + 1
			index = i + j
	#----case 3: handle trailing values
	end_values = arr[index]
	for i in range(index, end_of_arr):
		arr[i] = end_values
	return arr


# populates the evalutaion csv
# input: seq_len, number of seconds in a sequence
def populate_csv(h_m, h_sd, b_m, b_sd):

	s = "game frame, human mean, human stddev, bot mean, bot stddev\n"

	for i in range(len(h_m)):
		s += str(i) + ","
		s += str(h_m[i]) + ","
		s += str(h_sd[i]) + ","
		s += str(b_m[i]) + ","
		s += str(b_sd[i]) + "\n"

	f = open(EVAL_PATH, mode="w")
	f.write(s)
	f.close()
	exit()

	row1 = ['game frame']   + [len(h_m)]
	row2 = ['human mean']   + h_m
	row3 = ['human stddev'] + h_sd
	row4 = ['bot mean']     + b_m
	row5 = ['bot stddev']   + b_sd

	rows = [row1,row2,row3,row4,row5]

	with open(EVAL_PATH, mode='w') as evals:
		eval_writer = csv.writer(evals, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		# populates csv by row, to be transposed
		for row in rows:
			eval_writer.writerow(row)

	evals.close()

	# transpose data such that dimensions become
	# (24*seq_len + 1, 5)
	z_evals = zip(*csv.reader(open(EVAL_PATH, "rt")))
	csv.writer(open(EVAL_PATH, "wt")).writerows(z_evals)

# clear csv
def clear_past_evals():

    # Clears all csv eval entries
    file = open(EVAL_PATH, mode='w+')
    file.close()


    #df = pd.read_csv('./files/control_data.csv', encoding='utf-8-sig')


simulate_realtime(60)








