import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

audio_path = 'speech2.wav'

# Step 2: Load the Audio File using Librosa
y, sr = librosa.load(audio_path, sr=None)

# Step 3: Compute the Mel spectrogram using a moving window
n_fft = 1024  # Length of the FFT window
hop_length = 512  # Number of samples between successive frames
n_mels = 128  # Number of Mel bands
window_type = 'hann'  # Type of window function

# Define a function to compute Mel spectrogram using a moving window
def compute_mel_spectrogram(y, sr, n_fft, hop_length, n_mels, window_type):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels,
        window=window_type
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram.T

# Compute the Mel spectrogram
log_mel_spectrogram = compute_mel_spectrogram(y, sr, n_fft, hop_length, n_mels, window_type)
print("Mel spectrogram shape:", log_mel_spectrogram.shape)

# Plot the Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram.T, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Log Mel spectrogram')
plt.show()

def median_filter(sequence, kernel_size=5):
    """
    Apply a median filter to a sequence.
    
    Args:
    sequence (np.ndarray): The input sequence of binary decisions.
    kernel_size (int): The size of the median filter kernel. Must be an odd integer.
    
    Returns:
    np.ndarray: The filtered sequence.
    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd integer"
    
    half_size = kernel_size // 2
    padded_sequence = np.pad(sequence, (half_size, half_size), mode='edge')
    filtered_sequence = np.zeros_like(sequence)
    
    for i in range(len(sequence)):
        filtered_sequence[i] = np.median(padded_sequence[i:i + kernel_size])
    
    return filtered_sequence


# Step 4: Scale the Features
scaler = StandardScaler()
log_mel_spectrogram_scaled = scaler.fit_transform(log_mel_spectrogram)

# Step 5: Train a Neural Network Classifier
# Example training data (features and labels)
# Replace these with actual training data
X_train = log_mel_spectrogram_scaled
y_train = np.random.randint(0, 2, size=(log_mel_spectrogram_scaled.shape[0],))

model = Sequential([
    Dense(128, activation='relu', input_shape=(n_mels,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 6: Predict the class probabilities for each frame
y_pred_prob = model.predict(log_mel_spectrogram_scaled)

# Convert probabilities to binary decisions (0 or 1)
threshold = 0.5
y_pred = (y_pred_prob >= threshold).astype(int)

# Print the original predicted classes in a single line
predicted_classes = ''.join(map(str, y_pred.flatten()))
print(f"Original Predicted classes: {predicted_classes}")

# Apply custom median filter to the predicted classes
y_pred_smoothed = median_filter(y_pred.flatten(), kernel_size=5)

# Print the smoothed predicted classes in a single line
predicted_classes_smoothed = ''.join(map(str, y_pred_smoothed.astype(int)))
print(f"Smoothed Predicted classes: {predicted_classes_smoothed}")

# Step 7: Visualize the Original and Smoothed Predicted Classes
# Create a time vector for the frames
time_frames = librosa.frames_to_time(np.arange(log_mel_spectrogram.shape[0]), sr=sr, hop_length=hop_length)

plt.figure(figsize=(14, 6))
plt.plot(time_frames, y_pred, label='Original Predicted Classes', drawstyle='steps-pre', alpha=0.6)
plt.plot(time_frames, y_pred_smoothed, label='Smoothed Predicted Classes', drawstyle='steps-pre', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Class')
plt.title('Original and Smoothed Predicted Foreground (1) and Background (0) Classes Over Time')
plt.ylim(-0.1, 1.1)
plt.yticks([0, 1])
plt.legend()
plt.show()
