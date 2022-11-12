#Code written by Jason Whitmore

#importing standard machine learning libraries (and version numbers)

#v 1.14.5
import numpy as np

#v 1.10.0
import tensorflow as tf

#v 2.2.4
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LeakyReLU, Input, LSTM, Masking, Add

import os
import sys

#v 0.25.3
import pandas as pd


#v 0.21.3
import sklearn
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold

import time



#Hyperparameters for the predictive model

#The size of the input vector. Dependent on input .csv files
INPUT_SIZE = 4


#Number of "Neurons" in the hidden layers
UNITS_PER_HIDDEN_LAYER = 64



#Training hyperparameters

#Defines the "stepsize" when doing gradient descent
LEARNING_RATE = 1e-4

#The number of samples used in one gradient update. Higher number leads to more stable training
MINIBATCH_SIZE = 64

#The number of full passes over the dataset during training. Higher numbers can produce better models.
NUM_EPOCHS = 8

#Gradient clipping value (clips gradient value elementwise)
CLIP = 0.1

#Sequence length, in seconds
SEQUENCE_LENGTH = 60

USE_RESNET = False


#The filepath needed to find the existing pretrained detector model.
MODEL_PATH = "detector_model" + str(SEQUENCE_LENGTH) + ".h5"






class detector:
    
    def __init__(self):
        """
        Constructor for the model. Loads an existing model if possible.
        """

        #print debug info
        print("Numpy version: " + np.__version__)
        print("Tensorflow version: " + tf.__version__)
        print("Keras version: " + keras.__version__)

        #class member for the detector
        self.detector_model = self.create_LSTM_detector()
        self.stateful_detector_model = self.create_stateful_LSTM_detector()

        #Load model if it already exists
        if os.path.exists(MODEL_PATH):
            print("Loading existing model trained for " + str(SEQUENCE_LENGTH) + " seconds.")
            self.detector_model = keras.models.load_model(MODEL_PATH)
            self.stateful_load_weights()
        else:
            self.detector_model = self.create_LSTM_detector()




    def create_stateful_LSTM_detector(self):
        """
        Creates the recurrent (LSTM) predictive model that takes in single frames at a time.
        Used for the real time detector

        Returns the detector as a keras model
        """

        model = Sequential()

        model.add(Masking(mask_value=-1.0, batch_input_shape=(1, None, INPUT_SIZE)))
        #Adding the first hidden layer. This is a recurrent (LSTM) layer that can interpret sequences of inputs
        model.add(LSTM(UNITS_PER_HIDDEN_LAYER, return_sequences=True, stateful=True))
        
        #Adding the second hidden LSTM layer.
        model.add(LSTM(UNITS_PER_HIDDEN_LAYER, return_sequences=False, stateful=True))

        #The output layer. A sigmoid activation ensures values are in the range of (0,1)
        model.add(Dense(1, activation='sigmoid'))

        #Defining the optimizer for the training algorithm to use. Adam is a common optimizer.
        opt = keras.optimizers.Adam(lr=LEARNING_RATE, clipvalue=CLIP)

        model.compile(optimizer=opt, loss="binary_crossentropy")


        return model

    def create_LSTM_detector_resnet(self):

        input_layer = Input(shape=(None, INPUT_SIZE))

        mask_layer = Masking(mask_value=-1.0)(input_layer)

        lstm_1 = LSTM(UNITS_PER_HIDDEN_LAYER, return_sequences=True)(mask_layer)

        lstm_2 = LSTM(UNITS_PER_HIDDEN_LAYER, return_sequences=True)(lstm_1)

        add_layer = Add()([lstm_1, lstm_2])

        lstm_3 = LSTM(UNITS_PER_HIDDEN_LAYER, return_sequences=False)(add_layer)


        output_layer = Dense(1, activation="sigmoid")(lstm_3)



        opt = keras.optimizers.Adam(lr=LEARNING_RATE, clipvalue=CLIP)

        m = Model(input=input_layer, output=output_layer)

        m.compile(opt, loss="binary_crossentropy")

        return m


    def create_LSTM_detector(self):
        """
        Creates the recurrent (LSTM) predictive model that takes in a sequence of game states as input.

        Returns the detector as a keras model
        """

        if USE_RESNET:
            return self.create_LSTM_detector_resnet()

        model = Sequential()

        model.add(Masking(mask_value=-1.0, input_shape=(None, INPUT_SIZE)))
        #Adding the first hidden layer. This is a recurrent (LSTM) layer that can interpret sequences of inputs
        model.add(LSTM(UNITS_PER_HIDDEN_LAYER, return_sequences=True))
        
        #Adding the second hidden LSTM layer.
        model.add(LSTM(UNITS_PER_HIDDEN_LAYER, return_sequences=False))

        #The output layer. A sigmoid activation ensures values are in the range of (0,1)
        model.add(Dense(1, activation='sigmoid'))

        #Defining the optimizer for the training algorithm to use. Adam is a common optimizer.
        opt = keras.optimizers.Adam(lr=LEARNING_RATE, clipvalue=CLIP)

        #Compile the model using "Mean absolute error", which makes the training statistics easy to read.
        model.compile(optimizer=opt, loss="binary_crossentropy")

        return model
    
    def predict_stateful(self, frame):
        """
        Makes a prediction on a single frame of input.

        frame: A 1 dimensional numpy array of the game frame information

        Returns a float between (0,1) with 1 being a bot, 0 begin human.
        Returns -1 on bad input frame size
        """

        if frame.ndim != 1:
            print("frame input is not a 1d array, but has " + str(frame.ndim) + " dimensions")
            return -1
        
        if len(frame) != INPUT_SIZE:
            print("frame input is not of length 4, but is of length " + str(len(frame)))
            return -1
            

        return self.stateful_detector_model.predict(x=np.array([[frame]], dtype="Float32")).flatten()[0]
    

    def predict_realtime(self, csv_path):
        """
        Makes predictions using the stateful prediction model in order to simulate a real time
        detector use on an already parsed game.

        csv_path: The path to the csv to load.

        Returns a 1d numpy array of size 24 * sequence length containing the simulated
        real time predictions or None if the csv doesn't have enough data to work with

        Returns None if the csv path doesn't exist
        """

        #Check CSV path to see if it exists
        if not os.path.exists(csv_path):
            print("Specified csv file does not exist: " + csv_path)
            return None


        #Load the game into a csv
        input_array = pd.read_csv(csv_path).values

        #Slice off the top row that contains string labels
        #input_array = input_array[1:][:]

        #Find the index to cut off excess rows

        index = detector.calculate_end_index(input_array)

        if index == -1:
            #Not enough data to fulfill sequence length requirements
            print("File " + csv_path + " does not contain enough data. Skipping...")
            return None
                    
        input_array = input_array[0:index + 1][:]

        #Rescale the inputs
        input_array = detector.scale_inputs(input_array)

        
        ret = self.predict_stateful_game(input_array)

        return ret




    def predict_stateful_game(self, game_array):
        """
        Performs the stateful predict function on each frame in the game array.

        game_array: 2d array containing all the game state info. Should be preprocessed
        beforehand (scaling, slicing).

        Returns a 1d numpy array with the a length of 24 * S, where S is the sequence length in seconds.
        The index is the frame number, and the value at that index is the prediction in (0,1)

        Values of -1 indicate no prediction was made since no frame took place on that frame number.
        These missing frames will need to be processed further to "fill in the gaps"

        Returns None if the game_array is poorly shaped.
        """

        if game_array.ndim != 2:
            print("game_array does not have 2 dimensions. Instead it has " + str(game_array.ndim))
            return None

        if len(game_array[0]) != INPUT_SIZE:
            print("game_array has the wrong number of features per row")
            return None

        self.stateful_reset()

        r = np.zeros((24 * SEQUENCE_LENGTH))

        for i in range(len(r)):
            r[i] = -1

        for i in range(len(game_array)):
            #Determine the game frame number by rescaling the first vector component
            frame_number = int(game_array[i][0] * 24.0 * SEQUENCE_LENGTH)

            pred = self.predict_stateful(game_array[i])
            r[frame_number - 1] = pred

        self.stateful_reset()

        return np.array(r, dtype="Float32")

    def predict_stateful_allgames(self, csv_folder, sequence_length=SEQUENCE_LENGTH):
        """
        Makes stateful predictions across all the valid games in the csv folder 

        csv_folder: The folder containing the csv files with game input frames

        sequence length: The amount of game frames to use, in seconds from start of the game

        Returns a list of 1d numpy arrays containing predictions
        Returns None if the csv_folder is not valid
        """

        if not os.path.exists(csv_folder):
            print("path to folder: " + csv_folder + " does not exist")
            return None

        input_data, output_data = self.load_data(csv_folder)

        output_data = None

        r = []

        #For each game, run through the stateful predict

        for i in range(len(input_data)):

            a = self.predict_stateful_game(input_data[i])

            r.append(a)
        
        return r

    def stateful_graph(self, csv_folder, sequence_length=SEQUENCE_LENGTH, result_csv_path="./", bot_result_file="bot_stateful_result.csv", human_result_file="human_stateful_result.csv"):
        """
        Loads in all the valid game data from the csv folder, then runs the stateful predict function
        on each game. These results are then grouped into bot or human predicts. Then, the results are
        scaled so that all the arrays in each group are of equal length. These arrays are then element
        wise averaged and their standard deviations calculated. These two columns are then written to
        to separate result csvs for display.

        csv_folder: The folder containing the csvs of all the games

        sequence_length: The max length, in seconds, of game data input to use

        result_csv_path: The path to the folder where the results will be placed as .csv files

        bot_result_file: The filename of the csv that will output for all bot predictions

        human_result_file: The filename of the csv that will output for all human predictions
        """

        #Load the proper model

        #Make predictions
        all_games = self.predict_stateful_allgames(csv_folder, sequence_length=SEQUENCE_LENGTH)

        bot_games = []
        human_games = []

        for i in range(len(all_games)):
            if all_games[i][len(all_games[i]) - 1] >= 0.5:
                #bot
                bot_games.append(all_games[i])
            else:
                #human
                human_games.append(all_games[i])


        #May have to fix this stuff later there ends up being a more "correct" way of doing this
        array_size = 1000
        progress = np.arange(array_size)
        bot_prediction = np.zeros((len(bot_games), array_size))
        bot_mean = np.zeros(array_size)
        bot_std = np.zeroes(array_size)

        for i in range(len(bot_games)):
            for j in range(len(progress)):
                index = int(len(bot_games[i]) * (float(j) / len(progress)))
                bot_prediction[i][j] = bot_games[i][index]
            
        for i in range(len(bot_mean)):
            bot_mean[i] = np.mean(bot_prediction, axis=1)
            bot_std[i] = np.mean(bot_prediction, axis=1)

        
        human_mean = np.zeros(array_size)
        human_std = np.zeros(array_size)
        human_prediction = np.zeros((len(bot_games), array_size))

        for i in range(len(human_games)):
            for j in range(len(progress)):
                index = int(len(human_games[i]) * (float(j) / len(progress)))
                human_prediction[i][j] = human_games[i][index]
            
        for i in range(len(human_mean)):
            human_mean[i] = np.mean(human_prediction, axis=1)
            human_std[i] = np.mean(human_prediction, axis=1)

        human_result = np.zeros(array_size, 3)

        bot_result = np.zeros(array_size, 3)

        for r in range(len(human_result)):
            human_result[r][0] = r / float(array_size)
            human_result[r][1] = human_mean[r]
            human_result[r][2] = human_std[r]

            bot_result[r][0] = r / float(array_size)
            bot_result[r][1] = bot_mean[r]
            bot_result[r][2] = bot_std[r]

        #Write results to a csv file


    @staticmethod
    def np_to_csv(labels, array, filename):
        """
        Writes the numpy array to a csv file with the given labels in the first row

        labels: a list of strings that describe the column

        array: The 2d numpy array that will have it's contents written to the file

        filename: The name of the file which will be written to
        """

        s = ""

        for i in range(len(labels) - 1):
            s += labels[i] + ", "
        
        s += labels[-1] + "\n"

        for r in range(len(array)):
            for c in range(len(array[r]) - 1):
                s += str(array[r][c]) + ", "
            s += str(array[r][-1]) + "\n"

        f = open(filename, "w")
        f.write(s)
        f.close()


    def stateful_reset(self):
        """
        Resets the stateful model. Typically used before predicting from
        a new game.
        """
        self.stateful_detector_model.reset_states()

    def stateful_load_weights(self):
        """
        Loads the weights (parameters) from the stateless model to the
        stateful model.
        """
        self.detector_model.save_weights("weights.h5")
        self.stateful_detector_model.load_weights("weights.h5", reshape=True)



    def predict(self, game_csv_path, delimiter=","):
        """
        Makes a prediction on whether or not a player is a bot (0 is very unlikely, 1 is very likely)

        game_csv_path: The game data represented by a csv file containing information about the game "frames"

        delimiter: Character used to separate out the numbers in the input data file. Default is "," for easy use with
        .csv files

        Returns a float between (0,1) with 1 being bot, 0 being human. If the input csv is bad, then -1 is returned
        """

        if not game_csv_path.endswith(".csv"):
            print("Input game csv path does not include a .csv suffix")
            return -1

        if not os.path.exists(game_csv_path):
            print("Input game csv path does not exist: " + game_csv_path)
            return -1

        #Open the file, and read in the first (and only) line
        #input_array = np.genfromtxt(game_csv_path, delimiter=",")
        input_array = pd.read_csv(game_csv_path).values

        #Slice off the top row that contains string labels
        #input_array = input_array[1:][:]

        if input_array.ndim != 2 or input_array.shape[1] != INPUT_SIZE:
            print("Error: " + game_csv_path + " has incorrect dimensions: " + str(input_array.shape))
            return -1
        
        #determine the end of the sequence within the Sequence length boundary

        index = detector.calculate_end_index(input_array, sequence_length=SEQUENCE_LENGTH)

        if index == -1 or index == len(input_array):
            return -1

        input_array = input_array[0:index + 1][:]
        
        if input_array.ndim != 3:
            print("Error after sequence padding. ndim =  " + str(input_array.ndim))
            return -1
        
        #Make prediction with model
        return self.detector_model.predict(x=input_array).flatten()[0]



    @staticmethod
    def load_data(csv_folder, print_bad_csvs=False):
        """
        Loads the data found in the folder into a input and output numpy array.

        csv_folder: The folder where the csv data set resides. Recursively searches all folders

        Returns a tuple of type input (list of 2d numpy array) and a 1d numpy array of output labels
        Returns None if csv_folder does not exist
        """

        if not os.path.exists(csv_folder):
            print("csv_folder path does not exist: " + csv_folder)
            return None


        training_inputs = []
        training_outputs = []

        print("Loading data...")

        num_processed = 0

        lengths = []

        latest_start = 0

        num_human_removed = 0
        num_bot_removed = 0

        #Iterate through every file in the csv folder recursively
        for folder, directory, file_list in os.walk(csv_folder):

            #Iterate through every file name in the csv folder
            for file in file_list:

                #Since we need the full path, we need to join the folder path and file name
                full_path = os.path.join(folder, file)
                

                #Check to see if it's a csv file
                if ".csv" in file:
                    
                    #Convert the csv file to a numpy array.
                    #state_sequence = np.genfromtxt(full_path, delimiter=",")
                    state_sequence = pd.read_csv(full_path).values


                    if len(state_sequence) > 0 and state_sequence[0][0] > latest_start:
                        latest_start = state_sequence[0][0]
                        #print("New latest start: " + str(latest_start))
                        #print("New latest start file: " + file)

                    index = detector.calculate_end_index(state_sequence)
                    
                    if index > 0:
                        lengths.append(index)

                    if index == -1:
                        #Not enough data to fulfill sequence length requirements
                        if print_bad_csvs:
                            print("File " + file + " does not contain enough data. Skipping...")
                            
                        if "b" in file:
                            num_bot_removed += 1
                        else:
                            num_human_removed += 1

                    

                    
                    state_sequence = state_sequence[0:index + 1][:]

                    
                    #print("Max: " + str(m))

                    #Normalize the frame number s.t. the game start is 0 and ends at 1
                    state_sequence = detector.scale_inputs(state_sequence)



                    #Convert entries from 64 bit floats to 32 bit floats to save space.
                    state_sequence = np.array(state_sequence, dtype="Float32")


                    #Check for if any value is NaN, which will mess up the training
                    #If the current csv does contain NaN, dont include in training data
                    if not (np.isnan(state_sequence)).any():
                    
                        #Target output label
                        output_truth = 0

                        #Check if it's a bot and adjust label if needed
                        if "b" in file:
                            output_truth = 1
                        
                        #Add input and output data to lists

                        if len(state_sequence) > 0 and index > 0:

                            #Experimental: Add special tokens -2 to signify end of sequence
                            #for j in range(4):
                                #state_sequence[len(state_sequence) - 1][j] = -2.0
                            
                            training_inputs.append(state_sequence)
                            training_outputs.append([output_truth])
                            num_processed += 1

                        

                        
                        
                        #Alert user if a csv turns out to be anything other than a 2 dimensional table.
                        if state_sequence.ndim != 2:
                            print("Error: Input array is not 2 dimensional")
                            print(str(state_sequence.shape))
                            print(file)
                            exit(1)
                    
                    if num_processed % 1000 == 0:
                        print("Processed " + str(num_processed) + " games.")
                    

        #Print more debug information
        print("Data loaded. Average sequence length: " + str(np.mean(lengths)))
        print("Min, Max sequence lengths: (" + str(np.min(lengths)) + ", " + str(np.max(lengths)) + ")")
        print("Number of human games: " + str(training_outputs.count([0])))
        print("Number of bot games: " + str(training_outputs.count([1])))

        print("\nNumber of human games removed: " + str(num_human_removed))
        print("Number of bot games removed: " + str(num_bot_removed))
        
        #return_inputs = keras.preprocessing.sequence.pad_sequences(training_inputs, dtype="Float32", padding="post", value=0)

        return_outputs = np.array(training_outputs, dtype="Float32")

        return training_inputs, return_outputs
    
    @staticmethod
    def calculate_end_index(array, sequence_length=SEQUENCE_LENGTH):
        """
        Calculates the end index based on the sequence length that is provided

        array: The 2d numpy array with 4 columns, the first being the frame number

        Returns the frame index right before the frame number exceeds the sequence length
        """


        for i in range(len(array)):
            if array[i][0] * (1.0/24.0) > sequence_length:
                return i - 1

        return len(array) - 1


    def k_folds_training(self, csv_folder, folds=10):
        """
        Performs the industry standard machine learning training technique of K folds cross validation.
        This slices the training set into a specified number of folds, then trains the model, using one
        section as a test set.

        Resets the model after each fold. Saves model after training

        csv_folder: The folder to load data from

        folds: The number of folds to train over. Default of 10
        """

        #Load data

        all_training_inputs, all_training_outputs = detector.load_data(csv_folder)
        #Balance data by undersampling class with more data samples

        all_training_inputs, all_training_outputs = detector.undersample_data(all_training_inputs, all_training_outputs)

        num_human_games, num_bot_games = detector.calculate_class_count(all_training_outputs)

        print("Human game count:" + str(num_human_games))
        print("Bot game count: " + str(num_bot_games))

        #Permute the data

        indicies = np.arange(all_training_outputs.shape[0])

        np.random.shuffle(indicies)

        all_training_inputs = detector.subset_list(all_training_inputs, indicies)

        all_training_outputs = all_training_outputs[indicies]

        self.detector_model.summary()

        #Initialize metric lists
        all_f1 = []
        all_f2 = []
        all_precision = []
        all_recall = []
        all_loss = []

        kf = KFold(n_splits=folds)
        fold_num = 1

        for train_index, test_index in kf.split(all_training_inputs):

            #Reset model
            self.detector_model = self.create_LSTM_detector()

            print("Fold " + str(fold_num) + "/" + str(folds) + " started.")
            #train

            self.fit_model(detector.subset_list(all_training_inputs, train_index), all_training_outputs[train_index])

            self.detector_model.save(MODEL_PATH)
            pred = []
            #Generate outputs on test set

            all_training_inputs = keras.preprocessing.sequence.pad_sequences(all_training_inputs, dtype="Float32", padding="post", value=-1.0)

            for i in test_index:
                pred.append(self.detector_model.predict(np.array([all_training_inputs[i]])).flatten()[0])

            for i in range(len(pred)):
                pred[i] = round(pred[i])

            #Evaluate model on f1 score, f2 score, precision, recall, and loss
            f1 = fbeta_score(y_true=all_training_outputs[test_index], y_pred=pred, beta=1)
            f2 = fbeta_score(y_true=all_training_outputs[test_index], y_pred=pred, beta=2)
            precision = precision_score(y_true=all_training_outputs[test_index], y_pred=pred)
            recall = recall_score(y_true=all_training_outputs[test_index], y_pred=pred)
            loss = self.detector_model.evaluate(x=all_training_inputs[test_index], y=all_training_outputs[test_index], batch_size=4)


            all_f1.append(f1)
            all_f2.append(f2)
            all_recall.append(recall)
            all_precision.append(precision)

            all_loss.append(loss)

            print("Fold " + str(fold_num) + "/" + str(folds) + " completed.")
            print("Loss: " + str(loss))
            print("f1 score: " + str(f1))
            print("f2 score: " + str(f2))
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))

            print()

            fold_num += 1

            



        #Training complete. Save model and print out average of the metrics
        print("K folds training complete.")
        print("Average test f1 score across folds: " + str(np.mean(all_f1)))
        print("Average test f2 score across folds: " + str(np.mean(all_f2)))
        print("Average precision score across folds: " + str(np.mean(all_precision)))
        print("Average recall score across folds: " + str(np.mean(all_recall)))
        print("Average test loss across folds: " + str(np.mean(all_loss)))

        #Save trained model
        self.detector_model.save(MODEL_PATH)



    def fit_model(self, training_inputs, training_outputs, num_epochs=NUM_EPOCHS, minibatch_size=MINIBATCH_SIZE, verbose=True):
        """
        Custom training function that only pads sequences within minibatches.
        This greatly reduces the amount of memory which is required to train the model.

        training_inputs: The inputs to train the model on. Python list of 2d numpy arrays.

        training_outputs: The outputs used as training labels. Numpy array.

        num_epochs: The number of epochs to train the model for.

        minibatch_size: The number of data samples used in one parameter update.

        verbose: If true, then the epoch completed number, time to complete, and train loss will
        be displayed every epoch.

        """
        num_samples = len(training_inputs)

        for e in range(num_epochs):

            start = time.time()

            #Shuffle the training inputs and outputs
            
            indicies = np.arange(training_outputs.shape[0])

            np.random.shuffle(indicies)

            training_inputs = self.subset_list(training_inputs, indicies)

            training_outputs = training_outputs[indicies]



            samples_trained = 0

            epoch_losses = []

            while samples_trained < num_samples:

                #Slice out the current batch

                #make sure the batch index isn't out of bounds
                end_index = min(samples_trained + minibatch_size, len(training_inputs) - 1)

                batch_inputs = training_inputs[samples_trained : end_index + 1]

                if end_index - samples_trained + 1 <= 0:
                    break

                batch_outputs = training_outputs[samples_trained : end_index + 1]

                samples_trained += minibatch_size

                #Pad the current batch sequence
                batch_inputs = keras.preprocessing.sequence.pad_sequences(batch_inputs, dtype="float32", padding="post", value=-1.0)

                #train on batch
                loss = self.detector_model.train_on_batch(batch_inputs, batch_outputs)

                epoch_losses.append(loss)
                

                if len(epoch_losses) > 30:
                    epoch_losses.pop(0)
            
            if verbose:
                print("Epoch " + str(e + 1) + "/" + str(num_epochs) + " complete.")
                print("Average loss over epoch: " + str(np.mean(epoch_losses)))
                print("Time to complete epoch: " + str(time.time() - start))
                print()



    @staticmethod
    def subset_list(l, indicies):
        """
        Makes a subset list of a given list in the order of the indicies

        l: The list to make a subset out of.

        indicies: The indicies to put into the new subset. Order dependent.

        Returns the subset of the list
        """
        r = []

        for i in range(len(indicies)):
            r.append(l[indicies[i]])

        return r


    @staticmethod
    def scale_inputs(array):
        """
        Scales each component in a row such that the values stay in the range of [0,1]

        array: The 2d numpy array to scale

        Returns the scaled array
        """

        a = np.array(array, dtype="Float32")

        if len(a) == 0:
            return a

        

        for i in range(len(array)):
            a[i][0] = a[i][0] / (24.0 * SEQUENCE_LENGTH)

            a[i][1] = a[i][0] / 24.0

            a[i][2] = a[i][2] / 8192.0

            a[i][3] = a[i][3] / 8192.0
        
        return a

    def test_overfit(self, csv_folder, num_samples_list=[500, 1000, 1500, 2000], num_steps=5000, minibatch_size=MINIBATCH_SIZE, test_data_size=128):
        """
        Performs a black box test that intentionally overfits the model while training.
        Since the normal model itself is fairly large, we instead reduce the number
        of training samples to overfit, rather than use a large model.

        num_samples_list: List containing the number of sample counts to test.
        These numbers should be less than the total number of data samples available.

        test_data_size: Number of samples used for testing, rather than training.
        Returns None on failure
        """

        if len(num_samples_list) <= 0:
            print("num_samples_list does not contain enough entries")
            return None
        
        if num_steps <= 0:
            print("num_steps is invalid: " + str(num_steps))
            return None

        if minibatch_size <= 0:
            print("minibatch_size is invalid: " + str(minibatch_size))
            return None
        
        if test_data_size <= 0:
            print("test_data_size is invalid: " + str(test_data_size))
            return None

        if not os.path.exists(csv_folder):
            print("csv_folder does not exist: " + str(csv_folder))
            return None


        #Load the data
        input_data, output_data = self.load_data(csv_folder)

        #Shuffle data and isolate a test set before looking at

        x, y = detector.shuffle_data(input_data, output_data)

        test_x = x[0 : test_data_size]
        test_y = y[0 : test_data_size]

        train_x_total = x[test_data_size + 1:]
        train_y_total = y[test_data_size + 1:]

        for s in num_samples_list:

            print("Now training with " + str(s) + " samples...")
            
            #shuffle data
            x, y = detector.shuffle_data(train_x_total, train_y_total)


            train_x = train_x_total[0 : s + 1]
            train_y = train_y_total[0 : s + 1]


            #calculate the number of epochs to train for

            num_epochs = int((minibatch_size / float(len(train_x))) * num_steps)
            print("Train with " + str(num_epochs) + " epochs")

            #Reset the model to maintain a fair experiment
            self.detector_model = self.create_LSTM_detector()

            #fit the model
            self.fit_model(train_x, train_y, num_epochs, minibatch_size, verbose=False)

            all_test_loss = []
            all_train_loss = []

            #calculate test loss
            for i in range(len(test_x)):
                temp = np.array([test_x[i]], dtype="Float32")
                test_loss = self.detector_model.evaluate(x=temp, y=test_y[i], batch_size=1, verbose=0)
                all_test_loss.append(test_loss)

            #calculate train loss
            for i in range(len(train_x)):
                temp = np.array([train_x[i]], dtype="Float32")
                train_loss = self.detector_model.evaluate(x=temp, y=train_y[i], batch_size=1, verbose=0)
                all_train_loss.append(train_loss)

            print("Test loss = " + str(np.mean(all_test_loss)))
            print("Train loss = " + str(np.mean(all_train_loss)))
            print()



    def test_datashuffle(self, csv_folder, num_epochs=NUM_EPOCHS, minibatch_size=MINIBATCH_SIZE):
        """
        Performs a black box test which shuffles the training data so that each input
        is assigned a random output label. If the test is successful, the shuffled data
        should perform worse than the in order data.

        num_epochs: The number of epochs to train the model for.

        minibatch_size: The number of datasamples used in one parameter update.
        
        Returns None upon error
        """

        if num_epochs <= 0:
            print("Invalid number of epochs: " + str(num_epochs))
            return None
        
        if minibatch_size <= 0:
            print("Invalid minibatch size: " + str(minibatch_size))
            return None

        if not os.path.exists(csv_folder):
            print("csv_folder is invalid: " + csv_folder)
            return None

        train_x, train_y = self.load_data(csv_folder)

        #Train without shuffling first
        print("Training without data shuffling...")
        self.fit_model(train_x, train_y, verbose=True)


        #Train with shuffling
        indicies = np.arange(train_y.shape[0])

        np.random.shuffle(indicies)

        train_x = detector.subset_list(train_x, indicies)

        print("Training with data shuffling...")

        self.fit_model(train_x, train_y, verbose=True)


        

    @staticmethod
    def shuffle_data(input_data, output_data):
        """
        Shuffles the data rowwise, preserving the pair at any given index

        input_data: The input data to be shuffled. This is a python list.

        output_data: The output data to be shuffled. This is a numpy array.

        Returns the shuffled data as a (input, output) pair. Same types are the parameters
        """

        if len(input_data) != len(output_data):
            print("Length mismatch between input and output data.")
            return None

        indicies = np.arange(output_data.shape[0])

        np.random.shuffle(indicies)

        x = detector.subset_list(input_data, indicies)

        y = output_data[indicies]

        return x, y

    def calculate_avg_APM(self, csv_folder, label):
        """
        Calculates the average APM (actions per minute) of all games in the csv folder
        that match the label.

        csv_folder: The path to the folder containing the game csvs

        label: 0 for humans, 1 for bots

        Returns the APM as a floating point
        Returns -1 upon failure
        """
        if label == 0 or label == 1:
            print("label does not correspond to human (0) or bot (1): " + str(label))
            return -1

        if not os.path.exists(csv_folder):
            print("csv folder path does not exist: " + csv_folder)
            return -1

        x, y = self.load_data(csv_folder)

        apms = []

        for i in range(len(x)):
            if y[i] == label:
                
                unique_frame_numbers = set()
                for t in range(len(x[i])):

                    unique_frame_numbers.add(x[i][t][0])

                temp_apm = len(unique_frame_numbers) / (SEQUENCE_LENGTH / 60.0)

                apms.append(temp_apm)

        return np.mean(apms)



    @staticmethod
    def undersample_data(x, y):
        """
        Undersample (and rebalance) the data such that the human and bot datasets are of the same size by removing
        data from the larger set

        x: The input data, as a python list of 2d numpy arrays

        y: The output data, as a 1d numpy array

        Returns a tuple of the new undersampled data in the same form as the parameters
        Returns None if x and y's length do not match
        """

        if len(x) != len(y):
            print("x and y are of unequal length")
            return None

        num_human = 0
        num_bot = 0

        x_bot = []
        y_bot = []

        x_human = []
        y_human = []

        for i in range(len(y)):

            if y[i] == 0:
                num_human += 1

                x_human.append(x[i])
                y_human.append(y[i])
            else:
                num_bot += 1

                x_bot.append(x[i])
                y_bot.append(y[i])
        

        detector.shuffle_data(x_bot, np.array(y_bot))
        detector.shuffle_data(x_human, np.array(y_human))
            
        difference = abs(num_bot - num_human)

        if num_bot > num_human:

            while len(x_bot) > len(x_human):
                x_bot.pop(0)
                y_bot.pop(0)

        else:

            while len(x_human) > len(x_bot):
                x_human.pop(0)
                y_human.pop(0)

        ret_x = []
        ret_x.extend(x_bot)
        ret_x.extend(x_human)

        ret_y = []
        ret_y.extend(y_bot)
        ret_y.extend(y_human)

        ret_x, ret_y = detector.shuffle_data(ret_x, np.array(ret_y))

        return ret_x, np.array(ret_y)

    @staticmethod
    def calculate_class_count(y):
        """
        Determines how many data points belong to each class in the output data

        y: The output data (array of 1 or 0)

        returns a tuple in the form of num_human, num_bot
        """


        num_human = 0
        num_bot = 0

        for i in range(len(y)):

            if y[i] == 0:
                num_human += 1
            elif y[i] == 1:
                num_bot += 1
            else:
                print("Encountered target data element that is not 0 or 1: " + str(y[i]))
                return None

        return num_human, num_bot



    def test_unit_tests(self):
        """
        Runs unit tests on various function calls to see if desired behavior occurs

        Things commonly tested:

        -Invalid parameter range
        -Mismatched lists
        -Return values
        """

        print("Unit testing starting...")

        print("Testing load_data()")

        #Parameters to check: csv_folder path
        print("Testing a nonexistant csv folder path...")
        r = detector.load_data("folder_that_is_not_real")

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n\n")


        print("Testing predict_stateful()")
        print("Testing with poorly shaped 1d input array...")

        bad_array = np.array([0,0,0,0,0,0])

        r = self.predict_stateful(bad_array)

        if r == -1:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n")

        print("Testing with input array with too many dimensions...")

        bad_array = np.array([[0],[0]])

        r = self.predict_stateful(bad_array)

        if r == -1:
            print("Test passed.")
        else:
            print ("Test failed.")

        print("\n\n")


        print("Testing predict_realtime()")
        #One parameter to be tested: the csv location

        print("Testing a nonexistant csv path...")

        r = self.predict_realtime("filethatdoesnotexist")

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n\n")


        print("Testing predict_stateful_game()")
        #One parameter to test: game_array

        print("Testing a poorly shaped input array...")

        r = self.predict_stateful_game(np.zeros((1,2,3)))

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\nTesting an input array with the wrong number of columns")

        r = self.predict_stateful_game(np.zeros((4, 5)))

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n\n")

        
        print("Testing predict_stateful_allgames")
        #one parameter to test: csv_folder path

        print("Testing with a bad folder name...")

        r = self.predict_stateful_allgames("not_a_real_folder")

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n")

        print("Testing calculate_class_count()")
        #One input to test: y vector (should be 1d)

        print("Testing with a poorly shaped array")
        r = self.calculate_class_count(np.zeros((4,3)))

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing with an array containing number of than 0 or 1")

        r = detector.calculate_class_count(np.array([0,0,1,1,2]))

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n")


        print("Testing predict()")

        print("Testing with a nonexistent file")

        r = self.predict("not_a_real_file.csv")

        if r == -1:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing with a file that does not end with .csv")

        r = self.predict("not_a_real_file")

        if r == -1:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n")
        print("Testing undersample_data()")

        print("Testing with mismatched data sizes")

        r = detector.undersample_data([1,2,3,4], np.array([1,2,3]))

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing calculate_avg_APM()")

        print("Testing with a bad csv folder path")

        r = self.calculate_avg_APM("not_a_real_folder", 0)

        if r == -1:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing with a bad label")

        r = self.calculate_avg_APM("", 3)

        if r == -1:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n")
        

        print("Testing shuffle_data()")

        print("Testing with mismatched lengths of the input and output arrays")

        r = detector.shuffle_data(np.zeros((5)), np.array([1,2,3,4]))

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n")

        print("Testing test_datashuffle()")

        print("Testing with a bad csv_folder")

        r = self.test_datashuffle("bad_folder", 2, 2)

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing with bad epoch number")

        r = self.test_datashuffle("", -10, 2)

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing with a bad minibatch number")

        r = self.test_datashuffle("", 10, -5)

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("\n\n")

        print("Testing test_overfit()")

        print("Testing with a bad csv_folder path")

        r = self.test_overfit("bad_folder_here")

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing with a bad input list")

        r = self.test_overfit("", [])

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing with a bad num_steps")

        r = self.test_overfit("", [100, 200], -20)

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing with a bad minibatch size")
        r = self.test_overfit("", [100, 200], 20, -20)

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        print("Testing with a bad test sample size")

        r = self.test_overfit("", [100, 200], 20, 20, -150)

        if r == None:
            print("Test passed.")
        else:
            print("Test failed.")

        






        




        



        



        



