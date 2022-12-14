# IDD-Indian-Driving-Dataset- A Multi-label Problem

## Overview:
IDD consists of images, finely annotated with 16 classes collected from 182 drive sequences on Indian roads. 
The label set is expanded in comparison to popular benchmarks such as Cityscapes, to account for new classes. 
It also reflects label distributions of road scenes significantly different from existing datasets, with most classes displaying greater within-class diversity. 
Consistent with real driving behaviors, it also identifies new classes such as drivable areas besides the road. 
The dataset is inspired from the one used in the 2018 paper IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments

Source: 
http://idd.insaan.iiit.ac.in/dataset/details/

Splits:
The dataset of 9860 images has been split into train, validation and test sets with each set having 6,991, 1912 and 957 images, respectively.


# HOW TO DOWNLOAD DATA

1. pip install gdown

2. Go to the main folder - "IDD-Indian-Driving-Dataset", on the terminal, and execute following statements:

gdown --no-check-certificate https://drive.google.com/uc?id=1XOmsdCH9OAo2klfQ0o8D6tbPRxBWa4wV -O 'Data.zip'

3. Unzip files as follows:

unzip Data.zip

This will create a folder on your terminal named "Data".


# HOW TO RUN:

1. Please check Requirements.txt file to see additional packages that need to be installed.

2. Go to "Code" folder.

3. Simply execute the train.py file in the Code folder.


NOTE: The process takes around 3-4 hours to complete for 40 epochs.









