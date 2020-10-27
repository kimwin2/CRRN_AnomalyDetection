#CRRN (Convolutional Reconstructive Recurrent Network)

1. CRRN architecture
![Screenshot from 2020-10-27 16-46-18](https://user-images.githubusercontent.com/43340417/97271493-f83d3a80-1873-11eb-8c01-f0afc3c8ec69.jpg)

2. Usage
git clone https://github.com/kimwin2/CRRN_AnomalyDetection.git

for train, 
python3 0_preprocessing.py
python3 1_train.py --batch_size 5 --gpu_ids 0 --mode ConvConjLSTM

for test, 
python3 2_test_model.py --batch_size 5 --gpu_ids 0 --mode ConvConjLSTM



3. network files

ConvRRN.py
It is a wrapper class that adds functional things to use cstm.
SpatialAttention.py
This code is for spatial attention in the encoder-decoder structure.
core_module/ConvConjLSTM.py
This code is a cstm(convolutional spatio-temporal memory) structure.
core_module/ConvSTLSTM.py
This code is a convolutional spatio-temporal LSTM structure.
core_module/ConvLSTM.py
This code is a convolutional LSTM structure.


4. paper
Y.-H. Yoo, U.-H. Kim and J.-H. Kim, "Convolutional Recurrent Reconstructive Network for Spatiotemporal Anomaly Detection in Solder Paste Inspection," IEEE Trans. on Cybernetics, Accepted, Oct. 2020.


