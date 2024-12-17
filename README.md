# Emotion Detection from Audio using Machine Learning

## Project Overview
This project explores emotion detection from audio signals using machine learning and deep learning techniques. The primary goal is to classify human emotions (e.g., happy, sad, angry, etc.) based on audio recordings.

We focus on:
- Preprocessing audio signals into meaningful features (Mel spectrograms).
- Using Convolutional Neural Networks (CNNs) for feature extraction and classification.
- Evaluating and improving model performance across milestones.

## Repository Structure
This repository contains the implementation and results for Milestone 2 and Milestone 3.

### Files and Directories
```
|-- Milestone2.ipynb         # Jupyter Notebook for Milestone 2
|-- Milestone3.ipynb         # Jupyter Notebook for Milestone 3
|-- data/                    # Folder for audio dataset (CREMA-D)
|-- README.md                # Project documentation (this file)
```

### Dataset
We used the **CREMA-D** dataset, which contains labeled audio recordings of speakers expressing six emotions:
- **Happy**
- **Sad**
- **Angry**
- **Fear**
- **Disgust**
- **Neutral**

The dataset files are in `.wav` format, and spectrograms were generated during preprocessing.

## Milestone 2: Midway Checkpoint
### Key Highlights:
1. **Preprocessing**:
   - Audio data converted to **Mel spectrograms** using `librosa`.
   - Spectrograms resized to (128x128) for input to CNN.

2. **Model Architecture**:
   - Implemented a basic **Convolutional Neural Network (CNN)** using Keras.
   - Model Configuration:
     - Conv2D layers with filters: 32 and 64
     - MaxPooling and Flatten layers
     - Dense layers with ReLU and softmax activation

3. **Results**:
   - Validation Accuracy: **86.37%**
   - Confusion matrix and analysis indicate balanced classification across emotions.

4. **Challenges**:
   - Difficulty in extracting meaningful features from raw audio signals.
   - Limited accuracy improvements using classical ML models.

---

## Milestone 3: Final Model and Evaluation
### Key Improvements:
1. **Advanced Preprocessing**:
   - Improved spectrogram generation with data normalization.
   - Implemented **data augmentation** (time-stretching, pitch shifting).

2. **Enhanced Model**:
   - Tuned CNN architecture with:
     - Additional Conv2D layers (32, 64, 128 filters).
     - Dropout layers for regularization.
     - Learning rate adjustments.
   - Final CNN achieved **93.88% test accuracy**.

3. **Experiments with Other Architectures**:
   - **LSTMs/Bi-LSTMs**: Tested but yielded lower accuracy (~39%).
   - **Hybrid CNN+LSTM**: Limited improvements (~39%).
   - **Transformers**: Explored but resulted in poor performance due to insufficient tuning.

4. **Final Results**:
   - Test Accuracy: **93.88%**
   - Precision/Recall/F1: ~94% across all emotion classes.
   - Confusion matrix shows balanced performance across all labels.

### Key Takeaways:
- Spectrogram-based CNNs significantly outperform classical ML and raw data approaches.
- Data preprocessing and systematic model tuning are crucial for high accuracy.

---

## Running the Code
### Prerequisites:
1. Python 3.x
2. Required libraries:
   ```bash
   pip install numpy librosa keras tensorflow matplotlib
   ```

### Steps to Execute:
1. Clone the repository:
   ```bash
   git clone https://github.com/079035/Emotion-Classification/
   cd Emotion-Classification
   ```
2. Place the CREMA-D dataset in the `data/` folder.
3. Run the Jupyter Notebooks:
   - `Milestone2.ipynb`
   - `Milestone3.ipynb`

### Expected Outputs:
- Model training progress, validation accuracy, confusion matrix, and classification reports.

---

## Future Work
- Test the model on unseen datasets to evaluate robustness.
- Expand to more diverse datasets with additional emotions.
- Explore **transfer learning** using pre-trained audio models (e.g., Wav2Vec2).

## Team Members
- **Khushi Agarwal**
- **Richa Malhotra**
- **Jordi Del Castillo**

---

## Acknowledgments
- CREMA-D dataset providers
- Libraries: `librosa`, `Keras`, `TensorFlow`, `NumPy`

