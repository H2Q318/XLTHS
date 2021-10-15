from os.path import join
from scipy.io.wavfile import read
import numpy as np

FILE_PATH = "D:\Documents\XLTHS\BT nhom\TinHieuHuanLuyen"

def readFileInput(fileName):
    inputFile = []
    Voiced = []
    Unvoiced = []
    with open(join(FILE_PATH, fileName)) as file:
        for line in file:
            inputFile.append(line.split())
    for i in inputFile:
        Voiced.append(float(i[0]) + float(i[1]))
        Voiced.append(float(i[0]) - float(i[1]))
        Unvoiced.append(float(i[2]) + float(i[3]))
        Unvoiced.append(float(i[2]) - float(i[3]))
    print(Voiced)
    print(Unvoiced)
    return Voiced, Unvoiced

Voiced, Unvoiced = readFileInput("voiced_unvoiced.txt")
meanV = np.mean(Voiced)
stdV = np.std(Voiced)
meanU = np.mean(Unvoiced)
stdU = np.std(Unvoiced)
print(f"meanV = {meanV} stdV = {stdV} meanU = {meanU} stdU = {stdU}")

