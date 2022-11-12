#StarCraft: Brood War Bot Detector

##Overview

This folder of the project repository contains the code used to create and train a deep learning StarCraft: Brood War bot detector.

##Requirements

##Data

The training data for this model can be found here: https://drive.google.com/file/d/17IqB2mLw638mfUnVckmHwxOLwRU8nJeO/view?usp=sharing

Since the data takes up GBs of space, we decided it's best to host it separately.

Once unzipped, the contents should be placed in this folder, and the CSV_FOLDER path variable in the training_driver.py file should be changed to the folder location.

###Python dependencies

In order to use both detector.py and training_driver.py, there are a few external libraries that need to be installed. These are listed at the start of each file, but are copied here for convenience:

Python 3.6+

Numpy (v1.14.5)

TensorFlow (v1.10.0)

Keras (v2.2.4)

Pandas (v0.25.3)

Sklearn (v0.21.3)

It's highly recommended to use a python virtual environment to avoid messing up dependencies on other projects.

###System requirements
As with most machine learning tasks, the resources required to train the model can be quite demanding. A rough guide to the amount of resources required is listed below, as a function of the gameplay sequence length variable, in seconds.

| Sequence Length (seconds) | RAM (GBs) |
| ----------- | ----------- |
| 15 | 2.5 |
| 30 | |
| 45 | |
| 60 | 6.3 |
| 75 | |
| 90 | |
| 105 | |
| 120 | |
| 135 | |
| 150 | |
| 165 | |
| 180 | |

It's highly recommended that the smallest sequence length is used first so that you can get a feel for how much computing resources are used on the machine.



##General usage

The hyperparameters for the model and training can be found as global variables in all capital letters at the start of each file, just after the module import statements. These control most of the relevant hyperparameters that are used.

There are 3 functions that are meant to be called at a high level inside of the training_driver.py file: k_folds_training, test_datashuffle, and test_overfit.

###K-Folds cross validation training

This is the primary training function that is used in our project. It utilizes the industry standard K-Folds cross validation algorithm which not only trains the model, but also determines it's robustness and accuracy.

When called, the load_data() function will be called, which will read the data off the csvs that were stored on the disk. The user will be notified every 1000 games parsed. A warning message about lack of usable data will appear whenever a csv file doesn't contain valid game input frames given the specified sequence length. Such games will be disregarded and not loaded for training.

Once the data is loaded, the training will begin on a fold. Each epoch completed will output a training loss so that progress can be monitored. After all the epochs are finished for a fold, important experimental metrics will be calculated, such as test f1, f2, precision, and recall.

After all the folds are complete, all the test metrics from all folds will be averaged together and displayed.

##Tests

###Data shuffle

This black box test checks both the training code as well as the model performance. The idea behind this test is to train the model normally, keeping track of the train loss per epoch. Then, the labels for the training data (0 or 1) are then shuffled. The model is then trained on this shuffled data, keeping track of the train loss per epoch.

Ideally, the unshuffled data training should produce lower losses than the shuffled data training. The intution being that the optimization process should have a much more difficult time trying to find correlations in the data since the labels are random.


###Model overfitting

This black box test aims to intentionally overfit the model based on intuition surrounding model size versus amount of data.

In machine learning literature, the idea of overfitting a model often happens when the number of model parameters (model size) is too high for the amount of data used for training. If model size is used as an experimental variable and the number of data is kept constant, a "U" shaped test loss curve is typically seen. Ideally, researchers would pick the model size which is at the smallest value along this curve.

Our model, however, is already large enough, so it's infeasible to try and train on larger models for the sake of memory requirements. Instead, we can keep the model size the same, but instead vary the amount of data used for training.

The process starts by loading all of the training data. Then, a subset of this data is collected as test data. Anything not part of the test dataset is reduced to specified smaller groups. These groups are then trained on. After each group is trained, the test and train loss are plotted with the size of the group as the experimental variable.

The trend to expect is the test loss starting high, then converging with the train loss as the train loss stabilizes as the number of training data samples approaches the full data set.

