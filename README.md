# ASL Sign Language Recognition System
### Author: Archie Galbraith <@zanolablue>
This system consists of two primary components: a Convolutional Neural Network (CNN) model trained to recognize American Sign Language (ASL) signs from static images (asl_cnn_v2.py) and a live prediction script (ocv_asl_roi.py) that uses a webcam to capture video frames and predict signs in real-time using the trained model.

## Dependencies:
 1. Tensorflow
 2. Keras
 3. OpenCV

## 1. Model Training Script (asl_cnn_v2.py)
### Overview
The training script builds and trains a CNN to recognize ASL signs represented in static images. It processes the sign language dataset, applies data augmentation to improve generalization, and finally trains the CNN model.

### Key Features
Dataset Handling: Loads the ASL sign language dataset from CSV files, separating images and labels for training and testing.

Preprocessing: Converts images into the correct shape and normalizes pixel values.

Data Augmentation: Applies transformations like rotation, width shift, height shift, shear, zoom, and fill mode to augment the training data.
CNN Architecture: Constructs a sequential model with convolutional layers, max pooling, dropout for regularization and to prevent overfitting, and dense layers for classification.

Training: Trains the model with the augmented data, validating its performance using a separate test set (95% Test accuracy).

### Usage
Ensure you have the ASL dataset CSV files (sign_mnist_train.csv and sign_mnist_test.csv) accessible to the script. Run the script to train the model, then save as .keras (I used tensorflow.keras rather than stand-alone keras)

## 2. Live Prediction Script (ocv_asl_roi.py)
### Overview
This script leverages OpenCV to capture live video from a webcam, process frames in real-time, and use the pre-trained CNN model to predict ASL signs.

### Key Features
Model Loading: Loads the CNN model trained by asl_cnn_v2.py as a .keras file.

Real-Time Video Processing: Captures video frames from a webcam.

Region of Interest (ROI): Specifies an ROI within the video frame where the user should place their hand for sign recognition.

Image Preprocessing: Preprocesses frames from the ROI to match the input requirements of the CNN (resizing, grayscale conversion, normalization).

Prediction and Display: Predicts the ASL sign from the ROI and displays the predicted sign and certainty level on the video feed in real-time.

### Usage
Update the model path in ocv_asl_roi.py to point to the correct path for asl_cnn_v2.keras.
Run ocv_asl_roi.py, ensuring you have a webcam connected and accessible.
Place your hand within the specified ROI and perform ASL signs. The script will display the predicted sign and its certainty on the screen.


# How to run:
![Screenshot](Screenshot_2024-03-20_at_11.36.52_PM.png)

### 1. Download the scripts and save asl-cnn-v2.py as .keras type
### 2. Change file path in ocv_asl_roi.py model loading to correct path for .keras file
### 3. Use command line and run (e.g python ocv_asl_roi.py for Mac terminal)

