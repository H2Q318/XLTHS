from os.path import join
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time

FILE_PATH = "D:\Documents\XLTHS\BT nhom\TinHieuHuanLuyen"
PHONE_F1 = join(FILE_PATH, "phone_F1.wav")
PHONE_M1 = join(FILE_PATH, "phone_M1.wav")
STUDIO_F1 = join(FILE_PATH, "studio_F1.wav")
STUDIO_M1 = join(FILE_PATH, "studio_M1.wav")
FRAME_LENGTH = 0.02

def ACF(frequency, signal, begin):
    # Độ dài của frame (đơn vị mẫu)
    frameLength = int(FRAME_LENGTH * frequency)
    lagRange = frequency//450, frequency//70
    xx = []
    for lag in range(lagRange[0], lagRange[1]):
        x = [np.float64(i) for i in signal[begin : (begin + frameLength)]]
        xx.append(sum([x[m] * x[m + lag] for m in range(frameLength - lag)]))
    return xx

# Main
# Đọc tín hiệu mẫu
frequency, signal = read(PHONE_F1)
print("Frequency : ", frequency)
print("Signal : ", signal)
print(ACF(frequency, signal, 9200))
# Đồ thị
plt.plot(signal)
plt.ylabel("Amplitude")
plt.xlabel("Time (samples)")
plt.show()

# Khôi phục tín hiệu
sd.play(signal, frequency)
time.sleep(len(signal) / frequency)
sd.stop()