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
FRAME_LENGTH = 0.03

def readFileInput(fileName):
    inputFile = []
    with open(join(FILE_PATH, fileName)) as file:
        for line in file:
            inputFile.append(line.split())
    for i in range(0, len(inputFile) - 2):
        inputFile[i][0] = int(float(inputFile[i][0]) * frequency)
        inputFile[i][1] = int(float(inputFile[i][1]) * frequency)
    return inputFile

# Hàm tự tương quan
def ACF(signal, begin, frameLength, lagRange):
    xx = []
    for lag in range(lagRange[0], lagRange[1]):
        x = [np.float64(i) for i in signal[begin : (begin + frameLength)]]
        xx.append(sum([x[m] * x[m + lag] for m in range(frameLength - lag)]))
    xx = [i/xx[0] for i in xx]
    return xx

def timDinhCaoNhat(input):
    chopArray = []
    for i in range(1, len(input) - 1):
        if input[i] > input[i + 1] and input[i] > input[i - 1]:
            chopArray.append((i, input[i]))
    maxChop = max(chopArray, key=lambda x: x[1])
    return maxChop[0], maxChop[1]

def timNguong(input):
    Voiced = []
    Unvoiced = []
    input = input[:-2]
    j = 0
    for i in range(0, len(signal) - frameLength, frameLength // 2):
        if i < int(input[j][1]):
            ACFFrame = ACF(signal, i, frameLength, lagRange)
            if input[j][2] == 'v':
                Voiced.append(timDinhCaoNhat(ACFFrame)[1])
            elif input[j][2] == 'uv':
                Unvoiced.append(timDinhCaoNhat(ACFFrame)[1])
        else:
            j += 1

    meanV = np.mean(Voiced)
    meanU = np.mean(Unvoiced)
    stdV = np.std(Voiced)
    stdU = np.std(Unvoiced)
    print(f"meanV = {meanV} stdV = {stdV} meanU = {meanU} stdU = {stdU}")
    return meanV - stdV, meanV + stdV

def pitchContour():
    F0 = []
    for i in range(0, len(signal) - frameLength, frameLength // 2):
        ACFFrame = ACF(signal, i, frameLength, lagRange)
        indexChop, valueChop = timDinhCaoNhat(ACFFrame)
        if valueChop > nguong[0]:
            F0.append(frequency / indexChop)
        else:
            F0.append(0)
    F0 = np.array(F0)
    meanF0 = F0[np.nonzero(F0)].mean()
    stdF0 = F0[np.nonzero(F0)].std()
    print(f"meanF0 = {meanF0} stdF0 = {stdF0}")

    # j = 0
    # F0Temp = []
    # for i in range(0, len(signal) - frameLength, frameLength // 2):
    #     if i >= 9200:
    #         ACFFrame = ACF(signal, i, frameLength, lagRange)
    #         indexChop, valueChop = timDinhCaoNhat(ACFFrame)
    #         if valueChop > nguong[0]:
    #             F0Temp.append(frequency / indexChop)
    #         plt.plot(ACFFrame)
    #         plt.plot([0, frameLength], [0.3, 0.3])
    #         plt.plot(indexChop, valueChop, 'o')
    #         plt.title(f"F0Temp = {F0Temp[j]} Value = {valueChop}")
    #         j += 1
    #         plt.ylim([-1,1])
    #         plt.ylabel("Amplitude")
    #         plt.xlabel("Time (samples)")
    #         plt.show()
    return

# Main
# Đọc tín hiệu mẫu
frequency, signal = read(join(FILE_PATH, FILE_WAV[INDEX]))
print("Frequency : ", frequency)
print("Signal : ", signal)

# Độ dài của frame (đơn vị mẫu)
frameLength = int(FRAME_LENGTH * frequency)

# Độ trễ
# lagRange = frequency//450, frequency//70
lagRange = (0, frameLength)

# Tìm ngưỡng
nguong = timNguong(readFileInput(FILE_LAB[INDEX]))
# nguong = (0.3 , 1)
pitchContour()
