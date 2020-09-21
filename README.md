# Hands Off: Face Touch Detection Using Signal Processing

This repo holds the project code for Hands Off, a 6.808 final project, created by Alexa Jan and Joanna Cohen, in collaboration with the Saving Face project at the MIT Media Lab.

Here is a breakdown of the files in this repository:

data_motions/: contains the data used to train the NN and test the kalman and median filter (x represents a number in the following filenames)
- x_single_may5: data used to test the kalman filter in which the user was still at the beginning, moved slowly towards their face, and the stayed still at their face for a moment
- xm_single_may5: data used to train the NN in which the user touched their face one time
- notouch_face_single_x: data used to test the kalman filter and train the NN in which the user did not touch their face
- touch_face_singlex : data used to test the kalman filter and train the NN in which the user touched their face one time
- negm_single_may5: data used to train the NN in which the user did many movements that did not involve touching their face
- notouch_consecutive_may5: data used to test the kalman filter in which the user did many movements that did not involve touching their face

plots/: plots of the accelerometer and gryoscope data 

kalman_filter_v2: This is the kalman filter code that was used on the raspberry pi in real time, including the code that plays the sound when a face touching motion is detected. Final project version of the code.

median_filter: This file contains the median filter, which looks at the last w RSSI values and checks if the average of those values is above a threshold. This also contains a method that determines if there is a face touch just by checking if an RSSI value is above a threshold. This code works offline, but can be easily changed to work in real time similarly to the kalman filter. 

realtime_kalman: This is the code for the kalman filter that allowed it to work in real time by only using the past 15 RSSI values and associated accelerometer values. This file has an issue where, when a face touch was detected, it would alert of a face touch for multiple iterations. That issue was solved in kalman_filter_v2. (deprecated)

sensor_fusion: kalman filter not in real time - used to test the accuracy of the kalman filter using the data in data_motions/

smoothing: initial code used for kalman filter (deprecated) 
