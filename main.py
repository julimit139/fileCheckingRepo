## import
import numpy as np
import os
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns

## local import
import dataExtraction
import processing_func
import artifactDetection
import fileCreating

## PATHS
dataPath = 'D:\\TeleBrain\\Data\\PD_test_data\\'
#dataPath = 'D:\\TeleBrain\\Data\\PD_512Hz from 1_06_2020 to 31_03_2021\\'
#dataPath = 'D:\\TeleBrain\\Data\\temp\\'
descPath = 'D:\\TeleBrain\\Data\\Notes in Polish\\'
jsonPath = 'D:\\TeleBrain\\Data\\patient_frames\\'
resultsPath = 'D:\\TeleBrain\\results\\frame_results\\'
resultsWavePath = 'D:\\TeleBrain\\results\\frame_results_wave\\'


## EEG PARAMETERS
lower_freq = 1
upper_freq = 40
set_to_zero_threshold = 50e-6
channelsNames = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fz', 'Fp1', 'Fp2', 'O1', 'O2', 'Oz', 'P3', 'P4', 'T5', 'T6', 'Pz', 'T3', 'T4']
brain_waves = {
    "gamma": {"start": 26, "stop": 40},
    "beta": {"start": 13, "stop": 25},
    "alpha": {"start": 8, "stop": 12},
    "theta": {"start": 4, "stop": 7},
    "delta": {"start": 1, "stop": 3}
}

## PROCESSING PARAMETERS
# frame parameters
tag_name = 'Oczy zamknięte'
epoch_size = 8  # in seconds
overlap = int(epoch_size/2)  # in seconds
# filter parameters
butter_degree = 4  # TO_OPTIMIZE: find degree

## LOAD FOLDER
# load data list
fileList = os.listdir(dataPath)
# load description list
descList = os.listdir(descPath)

## INITIALIZE VARIABLES
patients_data = dict()  # contain json with eeg frames

## CREATE RESULT FOLDERS
if not os.path.isdir(jsonPath):
    os.makedirs(jsonPath, exist_ok=True)
    os.makedirs(resultsPath, exist_ok=True)
    os.makedirs(resultsWavePath, exist_ok=True)

### FOR EACH FILE
for file in fileList:
    print(file)

    ## EXTRACT SIGNAL
    print('reading data')
    data = dataExtraction.extractData(dataPath + file)

    # --- Here starts added code ---
    signals = data[0]
    output = artifactDetection.performEEPDetection(signals, data[4], data[1], data[2])
    # output = artifactDetection.performLFPDetection(signals, data[4], data[1], data[2], 0.625, data[2] / 2, 50)
    isArtifactList = output[0]
    array = fileCreating.markArtifacts(isArtifactList, data[2])
    fileCreating.createFile(array)
    # --- Here ends added code ---

    signals = np.transpose(data[0])

    ## EXTRACT SIGNAL PARAMETERS
    sampling_rate = data[2]

    ## GET DESCRIPTION DATA
    name = file.split('.')[0]
    descName = [x for x in descList if re.search(name, x)]

    ## FIND TIME STAMPS FOR TAGS IN DESCRIPTION DATA
    print('looking for tags in signal')
    tag_time = processing_func.get_time_tags(descName[0], tag_name, descPath)

    ## GET SIGNAL TO ANALYSIS
    if tag_time:

        ## GET FRAMES FROM SIGNAL
        print('extracting frames')
        patients_data = processing_func.get_tag_frames(name, signals, sampling_rate, epoch_size, overlap, tag_time)

        # store data in json
        print('saving frames into json')
        json_name = os.path.join(jsonPath, name+".json")
        out_file = open(json_name, "w")
        json.dump(patients_data, out_file)
        out_file.close()

        # if not processed open JSON file
        # print('loading frames from json')
        # with open(jsonPath + name+".json") as json_file:
        #     patients_data = json.load(json_file)

        ## GET SPECTRAL DATA FROM FRAME
        print('estimating peak frequency for brain waves')
        peak_results = processing_func.get_peak_results(name, patients_data, epoch_size, sampling_rate, channelsNames, brain_waves, butter_degree)
        # save results
        peak_results.to_csv(resultsWavePath + name + '.csv', index=True)

        ## PLOT SPECTRAL DATA
        print('plotting results')
        data_to_plot = peak_results[['channel', 'gamma', 'beta', 'alpha', 'theta', 'delta']]
        g = sns.pairplot(data_to_plot, hue='channel')
        g.savefig(resultsWavePath + name + "_hist.png")
        plt.close()

        ## basic statistics
        # result_main = data_to_plot.groupby([str('channel')]).mean()
        # print(result_main)
        # result_std = data_to_plot.groupby([str('channel')]).std()
        # print(result_std)

    else:
        print('no matching description file found')






