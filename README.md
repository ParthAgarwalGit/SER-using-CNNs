# SER-using-CNNs
Using the RAVDESS dataset to build a 2D Convolutional Neural Network (CNN) that can generalize across different voices, pitches, and genders.

Speech Emotion Recognition (SER) System
Author: Parth Agarwal
ID:2024ADPS0726P

📌 Project Overview
This project leverages Deep Learning to classify human emotions from vocal recordings. By transforming 1D audio waveforms into 2D Log-Mel Spectrograms, the system treats emotion detection as a pattern recognition task using a specialized Convolutional Neural Network (CNN).

The model is trained on the RAVDESS (Radio-Quebecois Audio-Visual Database of Emotional Speech and Song) dataset, specifically targeting 8 emotional states.

🎭 Classified Emotions
Neutral / Calm
Happy / Sad
Angry / Fearful
Disgust / Surprised


🛠️ Technical Pipeline
1. Data Preprocessing & Augmentation
To ensure the model learns robust emotional cues rather than specific actor voices, the following pipeline was implemented:
-Silence Trimming: Removed non-informative leading/trailing silence using librosa.
-Temporal Normalization: Padded/Truncated all samples to a fixed 3.0-second duration to maintain consistent CNN input dimensions.
-Feature Extraction: Converted raw signals into 128-band Mel Spectrograms, followed by a Power-to-DB conversion to match the logarithmic nature of human hearing.
-Augmentation (Training Set Only): Injected random white noise and performed pitch shifting to simulate diverse recording environments and vocal ranges.

2. CNN Architecture
The model utilizes a 2D CNN architecture optimized for spectral patterns:
-4 Convolutional Layers: Progressing from 32 to 256 filters to capture hierarchical textures (from simple pitch shifts to complex emotional harmonics).
-Regularization: Integrated Batch Normalization and Dropout (0.25 - 0.4) to combat overfitting on a small dataset.
-Global Average Pooling (GAP): Used in place of standard "Flatten" layers to reduce parameters and focus on spatial feature maps.

Classification Report

      emotion  precision    recall  f1-score   support

     Neutral       0.76      0.72      0.74        18
        Calm       0.86      0.94      0.90        33
       Happy       0.94      0.72      0.82        40
         Sad       0.79      0.75      0.77        40
       Angry       0.89      0.94      0.91        34
     Fearful       0.78      0.84      0.81        37
     Disgust       0.82      0.90      0.86        20
    Surprised      0.77      0.83      0.80        24

    accuracy                            0.83       246
    macro avg       0.83      0.83      0.83       246
    weighted avg    0.83      0.83      0.83       246

Bias Analysis (Gender Pitch)
A critical part of this study was evaluating "Pitch Bias." The model was tested against Male and Female cohorts separately to ensure fairness.
  Male Accuracy:   79.54%
  Female Accuracy: 81.47%
✅ Model is relatively balanced across genders.

🚀 Usage
1. Prerequisites
Install the required dependencies:
Bash
pip install tensorflow librosa numpy matplotlib seaborn scikit-learn

3. Live Inference
Run the predict.py script to analyze any unseen .wav file:
Bash
python predict.py "path/to/your_audio.wav"


📂 Repository Structure
AI_Club_Task.ipynb: Complete training, EDA, and evaluation notebook.

final_emotion_model.h5: Saved Keras model weights.

predict.py: Standalone inference script for external testing.

requirements.txt: List of required Python packages.
