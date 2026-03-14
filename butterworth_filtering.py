import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt



lmsFiltered_ppg_data = np.load(r"E:\\SRT\\lms_filtered_PPG_signal_1001-1030.npy", allow_pickle=True)
print("PPG data loaded successfully...")

idx = lmsFiltered_ppg_data[:, 0]
ppg_signal = lmsFiltered_ppg_data[:, 1] # numpy.ndarray of list of PPG signals
raw_ppg_signal = ppg_signal.copy()
N = len(ppg_signal)
fs = 20


b, a = butter(4, [0.5, 8], btype='bandpass', fs=fs)
for i in range(N):
    if i % 100 == 0:
        print(f"Processed {i} participants' PPG butterworth filtering...")
    if ppg_signal[i] is None or len(ppg_signal[i]) == 0:
        continue
    ppg_signal[i] -= np.mean(ppg_signal[i])
    ppg_signal[i] = filtfilt(b, a, ppg_signal[i])
print("PPG Butterworth filtering completed...")



n = len(ppg_signal[100])
frequencies = np.fft.rfftfreq(n, d=1/fs)
hamm = np.hamming(n)
ppg_analyzed = raw_ppg_signal[100]
filt_fft_values = np.fft.rfft(hamm * ppg_signal[100])
filt_power = np.abs(filt_fft_values)
fft_values = np.fft.rfft(hamm * ppg_analyzed)
power = np.abs(fft_values)  

plt.figure()
plt.plot(frequencies[:n//2+1], filt_power[:n//2+1], label='PPG Signal Frequency Domain', color='blue')
plt.plot(frequencies[:n//2+1], power[:n//2+1], label='raw PPG Signal Frequency Domain', color='orange', alpha = 0.5)
plt.ylim(0, max(filt_power)*1.1)
plt.legend()
plt.title("Butterworth Filtered PPG Signal Example")
plt.grid(True)
plt.show()

ppg_signal = np.array(ppg_signal, dtype=object)
ppg_signal_with_index = np.column_stack([idx, ppg_signal])
np.save(r"E:\\SRT\\butterworth_filtered_and_lms_filtered_PPG_signal_1001-1030.npy", ppg_signal_with_index)
