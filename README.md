#   Real-world Anomaly Detection in Surveillance Videos

This repository provides the implementation for the paper 'Real-world Anomaly Detection in Surveillance Videos' by Waqas Sultani, Chen Chen, Mubarak Shah.

## Abstract

The project aims to detect anomolous activities in surveillance videos. A pre-trained 3-D convolution network was used to generate input feature vectors and using multiple instance learning an artificial neural network was trained for classification. 

### Prerequisites
#### Dataset: 
UCF-Crime (http://crcv.ucf.edu/cchen/UCF_Crimes.tar.gz) courtesy of
Waqas Sultani. It is the original dataset used for the aforementioned paper.
#### Tools:
Caffe,
Facebook/C3D-1.0 (https://github.com/facebook/C3D),
Tensorflow,
Python

## Implementation Details
### PREPROCESSING:
Resize each video frame to 240*320 pixels and fix
frame rate at 30fps.
### FEATURE EXTRACTION:
C3D features for every 16-frame video clip
followed by l2 normalization. To obtain features
for a video segment, we take the average of all
16-frame clip features within that segment.
### TRAINING:
We input these features (4096D) to a 3-layer FC
neural network. The first FC layer has 512 units
followed by 32 units and 1 unit FC layers.
Using MIL we try to generate higher anomaly
score for anomalous videos than normal videos.

## Acknowledgments

* This project was only possible due the work done by Waqas Sultani, and his help during the course of this project.
* We are very gratefull to Dr. Rama Krishna Sai Gorthi, our academic advisor for the project.
* The inspiration behing the project was to look into the techniques for anomoly detection in videos and exploit such       techniqes to develop a real time automated moderator for surveillance.

## Contributers
* [Abhay Pratap Singh](https://github.com/abhay97ps)
* [Aditya Dhall](https://github.com/adi-dhal)

## Citation
* Sultani, Waqas, Chen Chen, and Mubarak Shah. "Real-world Anomaly Detection in Surveillance Videos." Center for Research in Computer Vision (CRCV), University of Central Florida (UCF) (2018).
