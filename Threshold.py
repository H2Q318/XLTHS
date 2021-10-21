from os.path import join
from scipy.io.wavfile import read
import numpy as np

FILE_PATH_THHL = 'D:\Documents\XLTHS\BT nhom\TinHieuHuanLuyen'
FILE_WAV_THHL = ['phone_F1.wav', 'phone_M1.wav', 'studio_F1.wav', 'studio_M1.wav']
FILE_LAB_THHL = ['phone_F1.lab', 'phone_M1.lab', 'studio_F1.lab', 'studio_M1.lab']

INDEX = 0
TIME_FRAME = 0.03

# def readFileInput(fileName):
#     inputFile = []
#     with open(join(FILE_PATH_THHL, fileName)) as file:
#         for line in file:
#             inputFile.append(line.split())
#     for i in range(0, len(inputFile) - 2):
#         inputFile[i][0] = int(float(inputFile[i][0]) * frequency)
#         inputFile[i][1] = int(float(inputFile[i][1]) * frequency)
#     return inputFile

def CalculateGeneralThreshhold(fileName):
    inputFile = []
    Voiced = []
    Unvoiced = []
    with open(join(FILE_PATH_THHL, fileName)) as file:
        for line in file:
            inputFile.append(line.split())
    for i in inputFile:
        # Voiced.append(float(i[0]) + float(i[1]))
        Voiced.append(float(i[0]) - float(i[1]))
        Unvoiced.append(float(i[2]) + float(i[3]))
        # Unvoiced.append(float(i[2]) - float(i[3]))
    print(Voiced)
    print(Unvoiced)
    return Voiced, Unvoiced

def getFramesArray(signal, frameLength):
    step = frameLength // 2
    frames = []
    index = 0
    for i in range(0, len(signal) // step):
        temp = signal[index : index + frameLength]
        frames.append(temp)
        index += step
    return frames

# Hàm tự tương quan
def ACF(frame, frameLength):
    N = len(frame)
    xx = np.zeros(N)
    frame = np.concatenate((frame, np.zeros(N)))
    for n in range(N):
        xx[n] = np.sum([frame[m] * frame[m + n] for m in range(frameLength - n)])
    xx = xx / np.max(xx)
    return xx

# Tìm ngưỡng
def findThreshold(framesACFArray, frequency, input):
    Voiced = []
    Unvoiced = []
    input = input[:-2]
    j = 0
    posSample = 0
    temp = []
    for i in range(len(framesACFArray)):
        peak = getHighestPeak(framesACFArray[i], frequency)
        if (peak + posSample) >= input[j][0] and (peak + posSample) <= input[j][1]:
            temp.append(framesACFArray[i][peak])
        else:
            if input[j][2] == 'v':
                Voiced.append((np.mean(temp), np.std(temp)))
            elif input[j][2] == 'uv':
                Unvoiced.append((np.mean(temp), np.std(temp)))
            j += 1
            temp = []
        posSample += len(framesACFArray[i]) // 2
    averageMeanV = [i[0] for i in Voiced]
    averageStdV = [i[1] for i in Voiced]
    averageMeanU = [i[0] for i in Unvoiced]
    averageStdU = [i[1] for i in Unvoiced]

    meanV = np.mean(averageMeanV)
    meanU = np.mean(averageMeanU)
    stdV = np.mean(averageStdV)
    stdU = np.mean(averageStdU)
    print(f"meanV = {meanV} stdV = {stdV} meanU = {meanU} stdU = {stdU}")
    return meanV, stdV, meanU, stdU

# Tìm đỉnh của 1 frame
def getHighestPeak(frame, frequency, f0Range = (70, 450)):
    sampleRange = (int(frequency / f0Range[1]), int(frequency / f0Range[0]))
    temp = frame[sampleRange[0] : sampleRange[1] + 1]
    return np.argmax(temp) + sampleRange[0]

# main
# Voiced = []
# Unvoiced = []
# for i in range(0, len(FILE_WAV_THHL)):
#     frequency, signal = read(join(FILE_PATH_THHL, FILE_WAV_THHL[i]))
#     frameLength = int(TIME_FRAME * frequency)
#     framesArray = getFramesArray(signal, frameLength)
#     framesACFArray = [ACF(framesArray[i], frameLength) for i in range(len(framesArray))]
#     meanV, stdV, meanU, stdU = findThreshold(framesACFArray, frequency, readFileInput(FILE_LAB_THHL[i]))
#     Voiced.append(meanV - stdV)
#     Unvoiced.append(meanU + stdU)
Voiced, Unvoiced = CalculateGeneralThreshhold("voiced_unvoiced.txt")
avgV = np.mean(Voiced)
avgU = np.mean(Unvoiced)
threshold = (avgV + avgU) / 2
print(f"Ngưỡng: {threshold}")

