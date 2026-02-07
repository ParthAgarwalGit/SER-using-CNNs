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

1. Data Augmentations
To prevent the model from memorizing specific actor voices (overfitting) and to increase the effective size of the training set, the following augmentations were applied only to the training data:

Noise Injection:
Method: Added random Gaussian white noise to the raw audio signal.
Factor: 0.005 (Adds a subtle hiss without drowning out the speech).
Purpose: Helps the model ignore background silence and focus on vocal energy.

Pitch Shifting:
Method: Shifted the pitch of the audio up using librosa.effects.pitch_shift.
Steps: +2 semitones.
Purpose: Simulates different vocal characteristics (e.g., making a male voice sound slightly higher), forcing the CNN to learn the pattern of the emotion rather than the specific pitch of the actor.

Note: Time Stretching was explored during EDA but excluded from the final training pipeline to maintain strict temporal alignment (3 seconds) for the CNN.

2. Key Settings & Hyperparameters
A. Audio Preprocessing
Sample Rate (SR): 22050 Hz (Standard for speech analysis; captures frequencies up to ~11kHz).
Silence Trimming: Top-dB = 20 (Removes leading/trailing silence quieter than 20dB below the peak).
Fixed Duration: 3.0 Seconds.
Shorter clips: Padded with zeros (silence) to reach 3s.
Longer clips: Truncated to the first 3s.

B. Feature Engineering (Spectrograms)
Feature Type: Log-Mel Spectrogram.
Mel Bands (Height): 128 (Vertical resolution).
Time Steps (Width): 130 (Horizontal resolution, resulting from 3s duration).
Scaling: Logarithmic (dB) scale (mimics human hearing sensitivity).
Final Input Shape: (128, 130, 1) (Grayscale Image).

C. Model Training Config
Batch Size: 64 (Optimized for GPU acceleration).
Optimizer: Adam (Adaptive Moment Estimation).
Initial Learning Rate: Default (0.001).
Loss Function: Sparse Categorical Crossentropy (Since targets are integers 0-7).
Epochs: 50 - 70 (Stopped early via callback).

D. Callbacks (Regularization)
EarlyStopping:
Monitor: val_loss.
Patience: 10 epochs (Stops if validation loss doesn't improve for 10 epochs).
ReduceLROnPlateau:
Monitor: val_loss.
Factor: 0.2 (Multiplies LR by 0.2 if stuck).
Patience: 5 epochs.
ModelCheckpoint: Saves only the weights with the highest val_accuracy.
