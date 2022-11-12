# Deep Learning for bot detection in StarCraft: Brood War

### By Dr. Michael Tsikerdekis, Sean Barret, Raleigh Hansen, Matthew Klein, Josh Orritt, and Jason Whitmore

This repository contains all the source code used for our research project as part of Western Washington University's CSCI 491/492/493 final project series. This project was overseen by Dr. Michael Tsikerdekis.

## Components

This project is split into 3 primary components that represent the workflow used to conduct the experiments in our research paper.

The primary programming language of this project was Python 3.6. Any external packages used are listed at the top of each python file with their version numbers

### Parser

This component was created by Sean Barret and Matthew Klein

Since there was no standardized StarCraft: Brood War dataset for bot detection, we were required to collect, parse, and clean the data ourselves. The code in this component converts StarCraft: Brood War replay files (.rep) into JSON files, then finally into comma separated value (.csv) files which can be used as time series data in our deep learning classifier model.


### Detector

This component was created by Jason Whitmore

Once the data has been parsed into .csv files, this component is used to construct and train deep learning models on this dataset.

Hyperparameters such has hidden layer size and sequence length are listed as variables in all capital letters at the start of the files. When running the game sequence length experiment, the model will be trained using K-fold cross validation and terminates with display of relevant statistics such as F1, F2, precision, and recall on the test sets.

### Evaluations

This component was created by Raleigh Hansen and Josh Orritt

Once a deep learning model is trained on our dataset, the evaluations component of our code provides tools to interpret and display results for the experiments we conducted. Notably, this component visualizes the simulated real time predictions experiment, which processes stateful predictions from each game, then plots a mean and standard deviation. These plots allow us to describe the accuracy and variance of our deep learning classifier as a game progresses.