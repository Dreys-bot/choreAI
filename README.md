# Dance Style Classification from 2D Poses
This project aims to develop a dance style recognition model from sequences of estimated 2D poses on videos.

## Context
Dance is a visual art form that uses the body and its movements as a means of expression. Being able to automatically analyze dance styles and techniques from videos would open up a wide range of pedagogical and cultural applications. This project proposes a method using pose estimation. It estimates 2D pose sequences, tracks dancers using AlphaPose with 17 key body points from an online gathered dataset to create temporal motion sequences. An LSTM recurrent neural network modelled the structure and numerous configurations were tested. Due to limited resources, the model achieved an accuracy of 62% with 24% error.

## Dataset
Several large dance video datasets have been extracted from YouTube, such as UCF-101, Kinectics and Atomic Visual Actions. However, these studies do not include the Afrobeat dance style. Therefore, it was necessary to create a custom dataset for this project, containing 1179 video clips equally distributed between the Afrobeat, classical and hip-hop styles. The videos were extracted from YouTube and TikTok and standardized in terms of duration (approximately 25 seconds) and resolution (Full HD).

## Pose Extraction
The AlphaPose model was used to estimate the 2D poses on each image of the videos. It detects 51 body landmarks, including 17 key points. The results were recorded in json files and then transformed into temporal series in text files.

## Model Training
A two-layer LSTM model was trained in a supervised manner on the pose sequences from the dataset. These were cut into fixed-size tensors to allow sequential processing by the model. The softmax cross-entropy cost function was used, and the Adam optimizer was employed to minimize the classification errors.

## Results and Future Work
The model achieved a maximum accuracy of 85% according to the learning rate. Planned improvements include estimating 3D poses, adding features other than poses, and generating new motion sequences.


## Installation
- Clone repository
  ```python
  git clone https://github.com/Dreys-bot/choreAI
  ```
- Install dependencies
```python
  pip install -r requirements.txt
  ```

## Run app 
```python
  python main.py
  ```
