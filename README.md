# LSTMLoc
# Dataprocessing
## Generatedata
Run the generatedata.py for generating original data from the log files.

## Overlapping
Please run the dataprocessing.py to generate training set data. Change the file path of overlap300/500/900 to change the overlapping rate.

# Training model
## Model Building
Please run the overlapping.py to run the downsample model. The downsample should be run after overlapping() function. which is already in the code. (Dont need to edit the code)

# Data visulisation tool
Run generateresults.py to visualise the sensor measurement from the raw log file to check if there is missing value in the raw file. 
