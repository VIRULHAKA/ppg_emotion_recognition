import numpy as np

'''
RUN THIS ONLY ONCE TO PREPROCESS THE RAW DATA INTO A UNIFORM FORMAT FOR MODEL TRAINING
'''

filteredPPG = np.load("E:\\SRT\\PPG&ACC_data\\butterworth_filtered_and_lms_filtered_PPG_signal_1001-1030.npy", allow_pickle=True)
# (764, 2) the second column is the valid PPG signal

rawGSR = np.load("E:\\SRT\\GSR_data\\GSR_with_v&a_data_1001-1030.npy", allow_pickle=True)
# [participant_id, esm_time, valence, arousal, physiol_data] * 764
# (764, 5) the last column is the valid GSR signal


vacantIndex = [0 if (filteredPPG[i, 1] is None or rawGSR[i, 1] is None) else 1 for i in range(len(filteredPPG))]
maskIndex = [i for i in range(len(vacantIndex)) if vacantIndex[i] == 1]
np.save(".\\dataset\\maskIndex.npy", maskIndex)
filteredPPG = filteredPPG[maskIndex, 1]
valence = rawGSR[maskIndex, 2]
arousal = rawGSR[maskIndex, 3]
rawGSR = rawGSR[maskIndex, 4]


N = len(maskIndex)
MAX_LEN = 36320

for i in range(N):
    if len(filteredPPG[i]) < MAX_LEN:
        filteredPPG[i] = np.pad(filteredPPG[i], (0, MAX_LEN - len(filteredPPG[i])), mode='constant')
    else:
        filteredPPG[i] = filteredPPG[i][:MAX_LEN]

    if len(rawGSR[i]) < MAX_LEN:
        rawGSR[i] = np.pad(rawGSR[i], (0, MAX_LEN - len(rawGSR[i])), mode='constant')
    else:
        rawGSR[i] = rawGSR[i][:MAX_LEN]

np.save(".\\dataset\\filteredPPG.npy", filteredPPG)
np.save(".\\dataset\\rawGSR.npy", rawGSR)
np.save(".\\dataset\\valence.npy", valence)
np.save(".\\dataset\\arousal.npy", arousal)