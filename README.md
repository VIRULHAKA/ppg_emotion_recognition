# ppg_emotion_recognition
Implemented a simple CNN for emotion recognition using PPG &amp; ACC data from dataset DAPPER. 

## Dataset and Signal Preprocessing
I used PPG and ACC data of participants 1001-1030 from dataset DAPPER. The PPG signals were first filtered by a bandpass butterworth filter. Then, I used a LMS adaptive filtering to reduce motion artifacts in PPG signals. 

## Model
A 1D CNN model implemented in Pytorch is used to extract features from processed PPG signals and perform 3-class emotion classification on valence and arousal.
