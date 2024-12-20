#Sign Language Detection Project#

##Overview

This project focuses on building a Sign Language Detection system using computer vision and machine learning. The system captures hand gestures through a webcam, processes them using MediaPipe, and classifies them into letters of the alphabet using a trained Random Forest model.

##Project Structure

collect_imgs.py: Script to collect sign language gesture images from a webcam and save them into class-specific folders.

train_classifier.py: Script to train a Random Forest classifier using collected image data.

inference_classifier.py: Real-time inference script to detect and classify hand gestures from webcam input.

##Features

###Data Collection: Collects 100 images per class for 26 classes (A-Z) using a webcam.

###Model Training: Trains a Random Forest classifier using image data.

###Real-time Inference: Detects and classifies hand gestures using MediaPipe.

##Dependencies

###OpenCV

###MediaPipe

###scikit-learn

###numpy

##Usage

###Data Collection: Run collect_imgs.py to collect sign language gesture images: python collect_imgs.py

###Model Training: Train the classifier using train_classifier.py: python train_classifier.py

###Real-time Inference: Start the inference script to detect sign language gestures:python inference_classifier.py

##How It Works

###Data Collection: Captures images of hand gestures for each letter of the alphabet.

###Preprocessing: Extracts hand landmarks using MediaPipe and normalizes coordinates.

###Training: Fits a Random Forest model using extracted features.

###Inference: Detects hand gestures from webcam input and predicts corresponding letters.

##Results

Model accuracy is displayed after training.

Real-time letter predictions are shown during inference.

##Future Improvements

Expand dataset for better model accuracy.

Implement more advanced models like CNNs.

Add support for dynamic sign language recognition.

##Acknowledgements

###MediaPipe for hand tracking.

###scikit-learn for machine learning.

###OpenCV for image processing.
