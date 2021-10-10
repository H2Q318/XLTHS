from os.path import join
from numpy import sign
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import sounddevice as sd
import time

FILE_PATH = "D:\Documents\XLTHS\BT nhom\TinHieuHuanLuyen"
PHONE_F1 = join(FILE_PATH, "phone_F1.wav")
PHONE_M1 = join(FILE_PATH, "phone_M1.wav")
STUDIO_F1 = join(FILE_PATH, "studio_F1.wav")
STUDIO_M1 = join(FILE_PATH, "studio_M1.wav")

# Đọc tín hiệu mẫu
frequency, signal = read(PHONE_M1)
print("Frequency : ", frequency)
print("Signal : ", signal)

# Đồ thị
plt.plot(signal)
plt.ylabel("Amplitude")
plt.xlabel("Time (samples)")
plt.title("Phone M1")
plt.show()

# Tính độ dài của file audio
duration = len(signal) / frequency

# Khôi phục tín hiệu
sd.play(signal, frequency)
time.sleep(duration)
sd.stop()