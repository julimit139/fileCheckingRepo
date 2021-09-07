import numpy as np
import dataExtraction
import artifactDetection
import fileCreating

dataPath = "D:\Data\PD_512\PD135.txt"
# dataPath = "D:\PD_512Hz from 1_06_2020 to 31_03_2021\PD148.txt"
# dataPath = "D:\PD_512Hz from 1_06_2020 to 31_03_2021\PD149.txt"
data = dataExtraction.extractData(dataPath)
signals = data[0];

output = artifactDetection.performEEPDetection(signals, data[4], data[1], data[2])
# output = artifactDetection.performLFPDetection(signals, data[4], data[1], data[2], 0.625, data[2]/2, 50)
isArtifactList = output[0]

array = fileCreating.markArtifacts(isArtifactList, data[2])
fileCreating.createFile(array)