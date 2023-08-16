# Early Signs of Heart Disease Prediction using 1D Convolutional Neural Networks

View notebook and in-depth walkthrough on [Kaggle](https://www.kaggle.com/code/henrytang05/ecg-detection-1d-cnn)

This repository contains the code and Jupyter Notebooks for a machine learning project that aims to predict early signs of heart disease using a combination of a 1D Convolutional Neural Network (CNN) and a parallel neural network, each respectively trained on Electrocardiogram (ECG) signals and patient metadata to achieve an approximate binary accuracy of 90%.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing and Scaling](#data-preprocessing-and-scaling)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Preventing Overfitting and Variance](#preventing-overfitting-and-variance)
- [Results](#results)
- [License](#license)

## Introduction

Heart disease is a leading cause of death globally, and early detection can significantly improve patient outcomes. With an approximate 54% median prediction accuracy of physicians in electrocardiograph readings, this project utilizes machine learning techniques, specifically Convolutional Neural Networks, aiming to improve detection of early signs of heart disease. The novelty lies in the dual approach of training parallel neural networks on ECG signals and patient metadata to achieve enhanced prediction accuracy.

## Dataset

The dataset used for this project comprises a diverse set of 21,000+ standard 12-Lead ECG signals and patient metadata from the [PTB-XL ECG dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset). It includes data collected from various sources with the majority annotations being confirmed by a second cardiologist, and is carefully curated to maintain privacy and ethical considerations. The dataset is split into training, validation, and test sets to ensure robust model assessment.

## Data Preprocessing and Scaling

Preprocessing and scaling play a crucial role in preparing the data for training. Use TensorFlow and Scikit-Learn for the following tasks:

1. **ECG Signal Preprocessing**: Normalize the ECG signals to have zero mean and unit variance using `sklearn.preprocessing.StandardScaler`. This helps the model converge faster and prevents issues due to varying signal scales.

## Model Architecture

The core architecture of this project involves two parallel neural networks:

1. **ECG Signal Network**: This network processes raw ECG signals using 1D Convolutional layers, capturing temporal patterns and features specific to heart health.

2. **Patient Metadata Network**: This neural network processes patient metadata, such as age, sex, and medical history, using fully connected layers to capture contextual information.

The outputs of these two networks are concatenated and fed into a fully connected layers for a final, more encompassing prediction.

## Training

1. Run the `preprocessing.ipynb` script to preprocess and store the data.
2. Run the `functionalModel.ipynb` script to load train the ECG & Patient Metadata Networks.
3. The trained weights will be saved in the `models` directory.

## Preventing Overfitting and Variance

To prevent overfitting and reduce variance in the model, consider implementing the following techniques:

1. **Dropout Layers**: Introduced dropout layers within the networks using `tf.keras.layers.Dropout` to randomly deactivate a fraction of neurons during training. This encourages the model to learn more robust features and reduces over-reliance on specific neurons.

2. **Batch Normalization**: Applied batch normalization using `tf.keras.layers.BatchNormalization` after convolutional layers. This technique helps stabilize training by normalizing the inputs to each layer and thus reduces internal covariate shift.

## Results

The combined model achieves an approximate binary accuracy of 90% on the test set, demonstrating its potential for early detection of heart disease.

## Changes made to improve accuracy

- Experiment with different network architectures to potentially improve prediction accuracy.
- Explore the incorporation of attention mechanisms to highlight relevant features in the ECG signals.
- Investigate additional patient data sources that could contribute to better predictions.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to extend, modify, or distribute this project following the terms of the license. If you have any questions or suggestions, please open an issue. Happy coding!
