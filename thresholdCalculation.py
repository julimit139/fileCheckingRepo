import math
import auxiliaryFunctions as aF


"""
Calculates thresholds for External Electrostatic Potentials (EEP) detection function.
Called inside the ``artifactDetection/detectEEP`` function. 
Parameters:
    minMaxList : list
        List of tuples containing minimum and maximum signal values from each time block of a channel.
Returns:
    thresholds : tuple 
        Tuple containing minimum and maximum thresholds values for channel on which detection function is called. 
"""


def calculateThresholdsEEP(minMaxList):
    # creating lists to contain normalised min and max values from all blocks
    minList = []
    maxList = []

    # iterating through minMaxList and filling minList and maxList with normalised values
    for (xmin, xmax) in minMaxList:
        xMin = xmin.item()
        xMax = xmax.item()
        if xMin != 0:
            xMinNorm = math.log(abs(xMin), 10)
            minList.append(xMinNorm)
        elif xMin == 0:
            minList.append(xMin)
        if xMax != 0:
            xMaxNorm = math.log(abs(xMax), 10)
            maxList.append(xMaxNorm)
        elif xMax == 0:
            maxList.append(xMax)

    # calculating median values in minList and maxList
    minMedian = aF.calculateMedian(minList)
    maxMedian = aF.calculateMedian(maxList)

    # calculating standard deviation values in minList and maxList
    minStandardDeviation = aF.calculateStandardDeviation(minList)
    maxStandardDeviation = aF.calculateStandardDeviation(maxList)

    # calculating values of minThreshold and maxThreshold
    minThreshold = minMedian + 6 * minStandardDeviation
    maxThreshold = maxMedian + 6 * maxStandardDeviation

    # creating tuple containing threshold values
    thresholds = (minThreshold, maxThreshold)

    # returning thresholds
    return thresholds


"""
Calculates threshold for potentials derived from ECG detection function.
Called inside the ``artifactDetection/performECGDetection`` function.
Parameters:
    None
Returns:
    threshold : float 
        Floating point threshold value. 
"""


def calculateThresholdECG():
    threshold = float(0.9)
    return threshold


"""
Calculates threshold for low-frequency potentials (LFP) detection function.
Called inside the ``artifactDetection/detectLFP`` function. 
Parameters:
    fourierList : list
        List containing Fourier-based function values from each time block of a channel.
Returns:
    threshold : float  
        Floating point threshold value for channel on which detection function is called. 
"""


def calculateThresholdLFP(fourierList):
    # calculating median value of fourierList
    median = aF.calculateMedian(fourierList)

    # calculating threshold value
    threshold = 0.75 + 0.25 * median

    # returning threshold
    return threshold
