# Real-time Body Language Recognition using Mediapipe and Machine Learning

This project demonstrates a real-time body language recognition system using Mediapipe for extracting body landmarks and various machine learning algorithms for classification. The main goal is to capture body posture and facial expressions through a webcam, process the data to extract meaningful features, and classify the body language using a pre-trained machine learning model.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Real-time Prediction](#real-time-prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project uses Mediapipe Holistic, an ML pipeline that provides features to detect human pose, face, and hand landmarks. We then use these landmarks to train machine learning models that can classify different body language states.

## Features

- Real-time capture and processing of webcam feed.
- Extraction of pose and facial landmarks using Mediapipe.
- Training and saving machine learning models.
- Predicting body language classes in real-time.
- Displaying predictions and probabilities on the video feed.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/BhanuSaketh/Body-Language-Decoder.git
    cd BodyLanguageDecoder
    ```

2. **Install required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain:
    ```text
    opencv-python
    numpy
    matplotlib
    mediapipe
    pandas
    scikit-learn
    ```

## Usage

1. **Prepare your dataset:**

    Ensure you have a CSV file (`coords.csv`) containing landmark coordinates and their corresponding classes. The dataset should have columns for each coordinate and a 'class' column for labels.

2. **Real-time prediction:**

    Run the script to start the webcam feed and make real-time body language predictions:

    ```bash
    python code.py
    ```

## Dataset

The dataset should be a CSV file (`coords.csv`) containing the coordinates of landmarks detected by Mediapipe, along with a 'class' column that contains the labels for different body language states.

## Real-time Prediction

The script `code.py` uses the webcam feed to capture real-time video, processes the frames to extract pose and facial landmarks using Mediapipe, and predicts the body language class using the pre-trained model. The predictions and probabilities are then displayed on the video feed.

## Results

The system displays the predicted body language class and its probability directly on the video feed, providing real-time feedback on body language recognition.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
