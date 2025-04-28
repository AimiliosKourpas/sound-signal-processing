---

# 🎵 Speech and Sound Signal Processing - Word Segmentation 🎵

###  University: University of Piraeus  
###  Department: Informatics  
###  Course: Speech and Sound Signal Processing  

---

## 🎯 Project Overview

In this project, we designed a system that takes a spoken sentence and segments it into words 🗣️✂️.  
The system detects when each word starts and ends, **without knowing in advance how many words** there are — only assuming a **small silence gap** between words.

Additionally, we created a program to **play each detected word** separately.  
Finally, the system **estimates the average pitch** (fundamental frequency) of the speaker.

---

## ⚙️ Classifiers Implemented

The following classifiers were trained and evaluated:

-  **Least Squares (LSQ)**
-  **Support Vector Machine (SVM)**
-  **Multilayer Perceptron (MLP)** (Three-layer neural network)
-  **Recurrent Neural Network (RNN)**

---

## ⚡ Important Rules

- Programming Language: **Python 3.12.4** 🐍
- ❗ No CNNs, no web services, no transfer learning allowed.
-  Deliverables: PDF documentation, source code (source2023.zip), auxiliary files (auxiliary2023.zip).

---

## 🔍 How It Works

The system does **binary classification**:  
✅ Speech (foreground) vs ❌ Non-speech (background).

**Main Steps:**
1. Extract **Mel spectrograms** 🎶 from sliding windows of the audio.
2. Classify each window as speech or non-speech.
3. Apply a **median filter** to smooth out small errors 🧹.
4. Find the **boundaries between words** based on the cleaned-up predictions 🧩.

---

## 🧠 Classifier Details

### 🖊️ Least Squares (LSQ) & Support Vector Machine (SVM)
- Simple models that output a continuous value.
- Then we threshold them to get binary speech/non-speech predictions.

### 🧮 Multilayer Perceptron (MLP)
- **Two hidden layers**: 128 and 64 neurons with ReLU activation ⚡.
- Output layer: Single neuron with **Sigmoid** activation 🧠.
- Trained with **Binary Cross-Entropy Loss**.

### 🔁 Recurrent Neural Network (RNN)
- Built using TensorFlow's **SimpleRNN** layers 🔁.
- Processes sequences of frames to capture **temporal dynamics**.
- Outputs one probability per time frame.

---

## 📂 Datasets

**Foreground (Speech)** 🗣️:  
- Common Voice Corpus Delta Segment 18.0 (as of 6/19/2024) 📚.

**Background (Noise / Non-speech)** 🔇:  
- ESC-50 dataset (Harvard Dataverse) 🎧.

(Selected only ~150 folders to keep things manageable.)

---

## 🛠️ Training

- 🧑‍🏫 **MLP**: trained with Dropout layers to prevent overfitting.
- 🛡️ **SVM**: trained with LinearSVC from scikit-learn.
- 🔢 **LSQ**: trained using simple matrix operations.
- 🔁 **RNN**: trained using SimpleRNN layers to model sequence data.

---

## 🧪 Testing and Evaluation

- 🎯 Tested on three WAV files: 5 seconds, 10 seconds, 20 seconds.
- 📜 Each test file has:
  - `.txt` file with ground-truth words.
  - `.json` file with ground-truth timestamps.

**Testing Process**:
1. Load the test WAV file.
2. Extract Mel spectrogram features.
3. Predict using all models.
4. Post-process with median filtering.
5. Detect speech segments.
6. Compare predictions to ground-truth annotations 📝.

---

## 📁 Project Structure

| 📄 File | 📜 Description |
|:---|:---|
| `train.py` | Training script for all models |
| `test.py` | Testing and evaluation script |

---

## 🛒 Libraries Used

- `os` 🗂️: File operations
- `numpy` ➗: Math operations
- `json` 📄: Handling annotation files
- `librosa` 🎶: Audio processing
- `joblib` 💾: Model saving/loading
- `scikit-learn` 📚: ML algorithms (SVM, preprocessing)
- `tensorflow.keras` 🤖: Neural networks (MLP, RNN)

---

## 🔥 Key Functions

### In `train.py`
- `load_train_audio_clips(limit=None)`: Load training audio.
- `extract_features(audio_clip)`: Get Mel spectrograms.
- `pad_features(features, expected_frames)`: Pad/truncate features.
- Train and save models (MLP, SVM, LSQ, RNN).

### In `test.py`
- Load audio and ground-truth.
- Predict frame-by-frame speech probability.
- Smooth with median filter.
- Detect segments and compare results.

---

## ✨ Final Thoughts

This project built a **speech segmentation system** that works **without prior word count knowledge**.  
It uses **traditional machine learning** and **simple RNNs**, without heavy neural network models like CNNs, or any external APIs 🌐🚫.

---
