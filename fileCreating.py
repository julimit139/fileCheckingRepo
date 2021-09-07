import os
import numpy as np

"""
Creates file containing start and end positions, expressed by samples, of blocks containing artifacts.
Parameters:
    array : ndarray
        Array containing start and end positions, expressed by samples, of blocks containing artifacts.
Returns:
    None
"""


def createFile(array):
    filePath = "result.txt"
    if os.path.exists(filePath):
        print("File already exists")
    else:
        with open("result.txt", "w") as file:
            file.write("Start\t")
            file.write("End\n")
            for row in array:
                file.write(str(row[0]) + "\t")
                file.write(str(row[1]) + "\n")
        file.close()


"""
Finds blocks containing artifacts in a list given by ``isArtifactList`` and calculates blocks positions. 
Parameters:
    isArtifactList : list 
        List of boolean values informing about artifact occurrence in each block.
    samplingRate : int
        Sampling rate used in EEG examination.
Returns:
    array : ndarray
        Array containing start and end positions, expressed by samples, of blocks containing artifacts.
"""


def markArtifacts(isArtifactList, samplingRate):
    # counting blocks containing artifacts
    arrayLength = isArtifactList.count(True)

    # creating an empty array
    array = np.zeros((arrayLength, 2), dtype=int)

    # finding blocks containing artifacts
    indexes = [i for i, x in enumerate(isArtifactList) if x == True]

    # initializing necessary variables
    arrayInd = 0
    blockDuration = 4  # for now, block length is 4 s
    step = blockDuration * samplingRate

    # calculating start and end position, expressed by samples, of a block containing artifact
    for ind in indexes:
        startPosition = ind * step
        endPosition = startPosition + step
        array[arrayInd, 0] = startPosition
        array[arrayInd, 1] = endPosition
        if arrayInd < arrayLength:
            arrayInd += 1

    return array
