# ASL Sign Language Recognition System
### Author: Archie Galbraith <@zanolablue>
This system consists of two primary components: a Convolutional Neural Network (CNN) model trained to recognize American Sign Language (ASL) signs from static images (asl_cnn_v2.py) and a live prediction script (ocv_asl_roi.py) that uses a webcam to capture video frames and predict signs in real-time using the trained model.

## Dependencies:
 1. Tensorflow
 2. Keras
 3. OpenCV

## 1. Model Training Script (asl_cnn_v2.py)
### Overview
This script builds and trains a CNN to recognize ASL signs represented in static images. It processes the sign language dataset, applies data augmentation to improve generalization, and tests the trained model.

### Key Features
Dataset Handling: Loads the MNIST ASL sign language dataset from CSV files, separating images and labels for training and testing.

Preprocessing: Converts images into the correct shape and normalizes pixel values.

Data Augmentation: Applies transformations like rotation, width shift, height shift, shear, zoom, and fill mode to augment the training data and train the model to preform better under real world testing.

CNN Architecture: Constructs a sequential model with convolutional layers, max pooling, dropout for regularization to prevent overfitting, and dense layers for classification.

Training: Trains the model with the augmented data, validating its performance using a separate test set (95% Test accuracy).

## 2. Live Prediction Script (ocv_asl_roi.py)
### Overview
This script leverages OpenCV to capture live video from a webcam, process frames in real-time, and use the pre-trained CNN model to predict ASL signs.

### Key Features
Model Loading: Loads the trained CNN model in .keras format.

Real-Time Video Processing: Captures video frames from a webcam.

Region of Interest (ROI): Specifies an ROI within the video frame where the user should place their hand for sign recognition.

Image Preprocessing: Preprocesses frames from the ROI to match the input requirements of the CNN and match training conditions (resizing, grayscale conversion, normalization).

Prediction and Display: Predicts the ASL sign from the ROI and displays the predicted sign and certainty level on the video feed in real-time.


# How to run:
![Screenshot](https://github.com/zanolablue/mnist-asl-cnn/blob/e5f27cf6bfdb17a02788777f07944cd80dbc6ee8/Screenshot%202024-03-20%20at%2011.36.52%20PM.png)

### 1. Download the scripts and save asl-cnn-v2.py as .keras type
### 2. Change file path in ocv_asl_roi.py model loading to correct path for .keras file 
### 3. Use command line to run OpenCV script (e.g 'python ocv_asl_roi.py' in UNIX terminal) - both scripts must be in same directory
### 4. Preform hand signs in the ROI rectangle to get live predictions with certainty level
### 5. Press 'q' to exit program
