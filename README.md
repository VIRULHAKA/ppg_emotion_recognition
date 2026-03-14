# ppg_emotion_recognition
Implemented a simple CNN for emotion recognition using PPG &amp; ACC data from dataset DAPPER. 

## Dataset and Signal Preprocessing
I used PPG and ACC data of participants 1001-1030 from dataset DAPPER. The PPG signals were first filtered by a bandpass butterworth filter. Then, I used a LMS adaptive filtering to reduce motion artifacts in PPG signals. 

## Model
A 1D CNN model implemented in Pytorch is used to extract features from processed PPG signals and perform 3-class emotion classification on valence and arousal.

## Training and Results
I used a WeightedRandomSampler to deal with the imbalance between the classes. Then I trained the model for 70 epochs. 
After training, the training loss decreased steadily from approximately 1.09 to 0.95, while the training accuracy improved from 36% to around 51%. However, the test loss fluctuated and gradually increased, indicating moderate overfitting. The test accuracy remained unstable between 15% to 45%, suggesting a limited generalization performance. 
