import math
import numpy as np
from scipy import stats
import auxiliaryFunctions as aF
import thresholdCalculation as tC


"""
Detects External Electrostatic Potentials (EEP) in channel given by ``channel``.
Parameters:
    channel : ndarray
        EEG channel, one of all EEG channels occurring in EEG examination data.
    examinationTime : int
        Duration of the EEG examination.
    samplingRate : int
        Sampling rate used in EEG examination.
Returns:
    isArtifact : list 
        List of boolean values informing about artifact occurrence in each block.    
    blockNumber : int
        Integer value informing about number of blocks in a channel data.
"""


def detectEEP(channel, examinationTime, samplingRate):
    # creating empty list for storing tuples containing minimum and maximum channel data values for each time block
    minMaxList = []

    # channel data is divided into many blocks where each block is 4 s long
    blockDuration = 4

    # number of blocks (integer, fractional parts are ignored)
    blockNumber = int(examinationTime / blockDuration)

    # step with which a block of channel data will be extracted
    step = blockDuration * samplingRate

    # indexes at which a block starts and ends - here the first block; they are incremented in the for loop
    startPosition = 0
    endPosition = startPosition + step

    # finding minimum and maximum channel data values in each block and filling minMaxList with them
    for block in range(blockNumber):
        xMin = min(channel[startPosition:endPosition])
        xMax = max(channel[startPosition:endPosition])
        minMaxList.append((xMin, xMax))
        startPosition += step
        endPosition += step

    # getting thresholds values for processed channel from function which calculates them
    thresholds = tC.calculateThresholdsEEP(minMaxList)
    minThreshold = thresholds[0]
    maxThreshold = thresholds[1]

    # creating and filling list with boolean values informing about artifact occurrence in each block
    isArtifact = []
    for (xmin, xmax) in minMaxList:
        xMin = xmin.item()
        xMax = xmax.item()
        if xMin != 0 and xMax != 0:
            if math.log(abs(xMin), 10) > minThreshold or math.log(abs(xMax), 10) > maxThreshold:
                isArtifact.append(True)
            else:
                isArtifact.append(False)
        elif xMin == 0 or xMax == 0:
            if xMin > minThreshold or xMax > maxThreshold:
                isArtifact.append(True)
            else:
                isArtifact.append(False)

    # returning list informing about artifact occurrences and integer value informing about number of blocks
    return isArtifact, blockNumber


"""
Performs detection function ``detectEEP`` on whole EEG examination data given by ``inputData``.
Parameters:
    inputData : ndarray
        Whole EEG examination data.
    eegChannelNumber : int
        Number of EEG channels in EEG examination data.
    examinationTime : int
        Duration of the EEG examination.
    samplingRate : int
        Sampling rate used in EEG examination.
Returns:
    isArtifactOutput : list 
        List of boolean values informing about artifact occurrence in each block of EEG examination data.   
    message : string
        Short text information about type of the artifact and it's occurrence in a block.
"""


def performEEPDetection(inputData, eegChannelNumber, examinationTime, samplingRate):
    # creating short information about type of the artifact and it's occurrence in a block
    message = "An artifact reflected by the external electrostatic potential occurrence has been detected in this " \
              "block"

    # creating empty list for storing information about artifact occurrence in each block of EEG data
    isArtifactOutput = []

    # performing EEP detection function in every channel and changing isArtifactOutput content respectively
    for channelNumber in range(eegChannelNumber):
        if eegChannelNumber == 19:
            channel = np.array(inputData[:, channelNumber + 1])
        elif eegChannelNumber == 20:
            channel = np.array(inputData[:, channelNumber])
        isArtifact = detectEEP(channel, examinationTime, samplingRate)[0]
        blockNumber = detectEEP(channel, examinationTime, samplingRate)[1]

        # filling isArtifactOutput with isArtifact content when iterating outer for loop for the first time
        if channelNumber == 0:
            isArtifactOutput = isArtifact

        # respectively changing or leaving as is the isArtifactOutput content when iterating outer for loop for second
        # and more times
        if channelNumber != 0:
            for block in range(blockNumber):
                if isArtifact[block]:
                    if isArtifactOutput[block] != isArtifact[block]:
                        isArtifactOutput[block] = isArtifact[block]

    # returning list informing about artifact occurrence in each block and message
    return isArtifactOutput, message


# **********************************************************************************************************************


"""
Detects potentials derived from ECG in data block given by ``dataBlock``.
Parameters:
    dataBlock : ndarray
        Part of EEG examination data of one time block and all examination data channels - both EEG and ECG.
    eegChannelNumber : int
        Number of EEG channels in EEG examination data.
Returns:
    maxCoefficient : float
        Maximum value in list of coefficients in a block.
"""


def detectECG(dataBlock, eegChannelNumber):
    # creating empty list fot storing correlation coefficient values for a block time and all channels
    coefficients = []

    # ndarray containing ECG signal values
    channelECG = np.array(dataBlock[:, 0])

    # calculating value of correlation coefficient of the signal in channel with ECG signal for all channels
    for channelNumber in range(eegChannelNumber):
        channel = np.array(dataBlock[:, channelNumber + 1])
        coefficient = stats.pearsonr(channelECG, channel)
        if coefficient[0] is not np.NaN:
            coefficients.append(coefficient[0])

    # finding maximum value in list of correlation coefficients in a time block
    if len(coefficients) > 0:
        maxCoefficient = max(coefficients)
    else:
        maxCoefficient = 0

    # returning maximum correlation coefficient
    return maxCoefficient


"""
Performs detection function ``detectECG`` on whole EEG examination data given by ``inputData``.
Parameters:
    inputData : ndarray
        Whole EEG examination data.
    eegChannelNumber : int
        Number of EEG channels in EEG examination data.
    examinationTime : int
        Duration of the EEG examination.
    samplingRate : int
        Sampling rate used in EEG examination.
Returns:
    isArtifactOutput : list 
        List of boolean values informing about artifact occurrence in each block of EEG examination data.
    message : string
        Short text information about type of the artifact and it's occurrence in a block.    
"""


def performECGDetection(inputData, eegChannelNumber, examinationTime, samplingRate):
    # creating short information about type of the artifact and it's occurrence in a block
    message = "An artifact derived from ECG has been detected in this block"

    # creating empty list for storing information about artifact occurrence in each block of EEG data
    isArtifactOutput = []

    # input data is divided into many blocks where each block is 4s long
    blockDuration = 4

    # number of blocks (integer, fractional parts are ignored)
    blockNumber = int(examinationTime / blockDuration)

    # step with which a block of input data will be extracted
    step = blockDuration * samplingRate

    # indexes at which a block starts and ends - here the first block; they are incremented in the for loop
    startPosition = 0
    endPosition = startPosition + step

    # getting threshold value from function which calculates it
    threshold = tC.calculateThresholdECG()

    # finding correlation coefficient maximum value in each block and all channels
    # and checking if an artifact occurs in a block
    for block in range(blockNumber):
        dataBlock = np.array(inputData[startPosition:endPosition, :])
        maxCoefficient = detectECG(dataBlock, eegChannelNumber)
        if maxCoefficient > threshold:
            isArtifactOutput.append(True)
        else:
            isArtifactOutput.append(False)
        startPosition += step
        endPosition += step

    # returning list informing about artifact occurrence in each block and message
    return isArtifactOutput, message


# *********************************************************************************************************************


"""
Detects low-frequency potentials (LFP) in channel given by ``channel``.
Parameters:
    channel : ndarray
        EEG channel, one of all EEG channels occurring in EEG examination data.
    examinationTime : int
        Duration of the EEG examination.
    samplingRate : int
        Sampling rate used in EEG examination.
    lambdaFrequency : float
        Lambda frequency value determined in EEGData class definition.
    nyquistFrequency : int
        Nyquist frequency value equal to half of sampling rate. 
    electricFrequency : int
        Electric network frequency determined in EEGData class definition.
Returns:
    isArtifact : list 
        List of boolean values informing about artifact occurrence in each block.    
    blockNumber : int
        Integer value informing about number of blocks in a channel data.
"""


def detectLFP(channel, examinationTime, samplingRate, lambdaFrequency, nyquistFrequency, electricFrequency):
    # creating empty list for storing Fourier based function values for each time block
    fourierList = []

    # channel data is divided into many blocks where each block is 4 s long
    blockDuration = 4

    # number of blocks (integer, fractional parts are ignored)
    blockNumber = int(examinationTime / blockDuration)

    # step with which a block of channel data will be extracted
    step = blockDuration * samplingRate

    # indexes at which a block starts and ends - here the first block; they are incremented in the for loop
    startPosition = 0
    endPosition = startPosition + step

    # finding Fourier-based function values in each block and filling fourierList with them
    for block in range(blockNumber):
        fourierValue = aF.calculateFourierFunction(channel[startPosition:endPosition], samplingRate, lambdaFrequency, nyquistFrequency, electricFrequency)
        fourierList.append(fourierValue)
        startPosition += step
        endPosition += step

    # getting threshold value from function which calculates it
    threshold = tC.calculateThresholdLFP(fourierList)

    # creating and filling list with boolean values informing about artifact occurrence in each block
    isArtifact = []
    for fourierValue in fourierList:
        if fourierValue > threshold:
            isArtifact.append(True)
        else:
            isArtifact.append(False)

    # returning list informing about artifact occurrences and integer value informing about number of blocks
    return isArtifact, blockNumber


"""
Performs detection function ``detectLFP`` on whole EEG examination data given by ``inputData``.
Parameters:
    inputData : ndarray
        Whole EEG examination data.
    eegChannelNumber : int
        Number of EEG channels in EEG examination data.
    examinationTime : int
        Duration of the EEG examination.
    samplingRate : int
        Sampling rate used in EEG examination.
    lambdaFrequency : float
        Lambda frequency value determined in EEGData class definition.
    nyquistFrequency : int
        Nyquist frequency value equal to half of sampling rate. 
    electricFrequency : int
        Electric network frequency determined in EEGData class definition.
Returns:
    isArtifactOutput : list 
        List of boolean values informing about artifact occurrence in each block of EEG examination data.
     message : string
        Short text information about type of the artifact and it's occurrence in a block.    
"""


def performLFPDetection(inputData, eegChannelNumber, examinationTime, samplingRate, lambdaFrequency, nyquistFrequency, electricFrequency):
    # creating short information about type of the artifact and it's occurrence in a block
    message = "An artifact reflected by the low-frequency potential occurrence has been detected in this block"

    # creating empty list for storing information about artifact occurrence in each block of EEG data
    isArtifactOutput = []

    # performing EEP detection function in every channel and changing isArtifactOutput content respectively
    for channelNumber in range(eegChannelNumber):
        if eegChannelNumber == 19:
            channel = np.array(inputData[:, channelNumber + 1])
        elif eegChannelNumber == 20:
            channel = np.array(inputData[:, channelNumber])
        isArtifact = detectLFP(channel, examinationTime, samplingRate, lambdaFrequency, nyquistFrequency, electricFrequency)[0]
        blockNumber = detectLFP(channel, examinationTime, samplingRate, lambdaFrequency, nyquistFrequency, electricFrequency)[1]

        # filling isArtifactOutput with isArtifact content when iterating outer for loop for the first time
        if channelNumber == 0:
            isArtifactOutput = isArtifact

        # respectively changing or leaving as is the isArtifactOutput content when iterating outer for loop for second
        # and more times
        if channelNumber != 0:
            for block in range(blockNumber):
                if isArtifact[block]:
                    if isArtifactOutput[block] != isArtifact[block]:
                        isArtifactOutput[block] = isArtifact[block]

    # returning list informing about artifact occurrence in each block
    return isArtifactOutput, message
