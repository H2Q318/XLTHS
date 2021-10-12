from os.path import join
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time

FILE_PATH = "D:\Documents\XLTHS\BT nhom\TinHieuHuanLuyen"
FILE_WAV = ["phone_F1.wav", "phone_M1.wav", "studio_F1.wav", "studio_M1.wav"]
FILE_LAB = ["phone_F1.lab", "phone_M1.lab", "studio_F1.lab", "studio_M1.lab"]
INDEX = 0
TIME_FRAME = 0.03

def readFileInput(fileName):
    inputFile = []
    with open(join(FILE_PATH, fileName)) as file:
        for line in file:
            inputFile.append(line.split())
    for i in range(0, len(inputFile) - 2):
        inputFile[i][0] = int(float(inputFile[i][0]) * frequency)
        inputFile[i][1] = int(float(inputFile[i][1]) * frequency)
    return inputFile

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
def findThreshold(framesACFArray, input):
    Voiced = []
    Unvoiced = []
    input = input[:-2]
    # j = 0
    # for i in range(0, len(signal) - frameLength, frameLength // 2):
    #     if i < int(input[j][1]):
    #         ACFFrame = ACF(signal, i, frameLength)
    #         if input[j][2] == 'v':
    #             Voiced.append(timDinhCaoNhat(ACFFrame)[1])
    #         elif input[j][2] == 'uv':
    #             Unvoiced.append(timDinhCaoNhat(ACFFrame)[1])
    #     else:
    #         j += 1

    j = 0
    for i in framesACFArray:
        if i[len(i) - 1] < int(input[j][1]):
            if input[j][2] == 'v':
                Voiced.append(np.max(i))
            elif input[j][2] == 'uv':
                Unvoiced.append(np.max(i))
        else:
            j += 1

    meanV = np.mean(Voiced)
    meanU = np.mean(Unvoiced)
    stdV = np.std(Voiced)
    stdU = np.std(Unvoiced)
    print(f"meanV = {meanV} stdV = {stdV} meanU = {meanU} stdU = {stdU}")
    # return meanV - stdV, meanV + stdV

def getPitch(frame, frequency, threshold = 0.6, f0Range = (70, 450)):
    sampleRange = (int(frequency / f0Range[1]), int(frequency / f0Range[0]))
    temp = frame[sampleRange[0] : sampleRange[1] + 1]
    posOfChop = np.argmax(temp) + sampleRange[0]
    if frame[posOfChop] > threshold:
        return frequency / posOfChop
    return 0

# Main
# Đọc tín hiệu mẫu
frequency, signal = read(join(FILE_PATH, FILE_WAV[INDEX]))
print("Frequency : ", frequency)
print("Signal : ", signal)

frameLength = int(TIME_FRAME * frequency) # Độ dài của frame (đơn vị mẫu)
framesArray = getFramesArray(signal, frameLength)
framesACFArray = [ACF(framesArray[i], frameLength) for i in range(len(framesArray))]
F0 = np.zeros(len(framesArray))
F0 = [getPitch(i, frequency) for i in framesACFArray]
F0mean = np.mean([i for i in F0 if i > 0])
F0std = np.std([i for i in F0 if i > 0])

findThreshold(framesACFArray, readFileInput(FILE_LAB[INDEX]))

# Show
plt.subplot(2, 1, 1)
plt.title("Signal")
plt.plot(signal)
plt.subplot(2, 1, 2)
plt.title(f"F0 - F0mean = {F0mean}, F0std = {F0std}")
plt.plot(F0, '.')
plt.tight_layout()
plt.show()
