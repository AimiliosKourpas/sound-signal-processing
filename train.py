import os
import numpy as np
import librosa
import joblib
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Input
from tensorflow.keras import layers

sr = 16000
n_mels = 128
expected_frames = 63

def load_train_audio_clips(limit=None):
    return load_foreground_train_audio_clips(limit), load_background_train_audio_clips(limit)

def load_foreground_train_audio_clips(limit=None):
    foreground_dir = os.path.join("datasets", "train", "foreground", "clips")
    foreground_train_audio_clips = [f for f in os.listdir(foreground_dir) if not f.startswith('.')]
    if limit:
        foreground_train_audio_clips = foreground_train_audio_clips[:limit]
    print(f"Loading {len(foreground_train_audio_clips)} foreground clips.")
    return [load_train_audio_clip(audio_name, "foreground") for audio_name in foreground_train_audio_clips]

def load_background_train_audio_clips(limit=None):
    background_dir = os.path.join("datasets", "train", "background", "clips")
    background_train_audio_clips = [f for f in os.listdir(background_dir) if not f.startswith('.')]
    if limit:
        background_train_audio_clips = background_train_audio_clips[:limit]
    print(f"Loading {len(background_train_audio_clips)} background clips.")
    return [load_train_audio_clip(audio_name, "background") for audio_name in background_train_audio_clips]

def load_train_audio_clip(audio_name, directory):
    audio_path = os.path.join("datasets", "train", directory, "clips", audio_name)
    try:
        audio_clip, _ = librosa.load(audio_path, sr=sr, offset=1.5, duration=2)
        return librosa.util.fix_length(audio_clip, size=sr * 2)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return np.array([])

def extract_features(audio_clip):
    if audio_clip.size == 0:
        return np.array([])

    mel_spectrogram = librosa.feature.melspectrogram( y=audio_clip, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram.T

def pad_features(features, expected_frames):
    if features.size == 0:
        return np.zeros((expected_frames, n_mels))
    if features.shape[0] < expected_frames:
        padding = np.zeros((expected_frames - features.shape[0], features.shape[1]))
        features = np.vstack((features, padding))
    return features[:expected_frames, :]

def train_mlp_classifier(train_features, train_labels):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(train_features.shape[1],)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_features, train_labels, epochs=20, batch_size=32, validation_split=0.2)
    model.save("mlp_classifier.h5")
    print("MLP classifier trained and saved successfully.")

def train_svm_classifier(train_features, train_labels):
    clf = LinearSVC(random_state=42, max_iter=10000)  # Increased max_iter
    clf.fit(train_features, train_labels)
    path = 'svm_classifier.pkl'
    joblib.dump(clf, path)

def train_lstsq_classifier(train_features, train_labels):
    X_train_bias = np.hstack([np.ones((train_features.shape[0], 1)), train_features])
    W = np.linalg.pinv(X_train_bias) @ train_labels  # Use pseudo-inverse to handle singular matrices
    path = 'lstsq_classifier.pkl'
    joblib.dump(W, path)

def train_rnn_classifier(train_features, train_labels):
    input_shape = (train_features.shape[1], train_features.shape[2])  # (n_frames, n_mels)
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(layers.SimpleRNN(32, return_sequences=True, kernel_initializer='he_normal'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(train_features, train_labels, epochs=50, batch_size=32, validation_split=0.15, verbose=0)
    model.save("rnn_classifier.h5")
    print("RNN classifier trained and saved successfully.")



foreground_train_audio_clips, background_train_audio_clips = load_train_audio_clips(limit=150)

print("Extracting features...\n")
foreground_train_features = np.array([pad_features(extract_features(audio_clip), expected_frames) for audio_clip in foreground_train_audio_clips])
background_train_features = np.array([pad_features(extract_features(audio_clip), expected_frames) for audio_clip in background_train_audio_clips])

foreground_train_features = [features for features in foreground_train_features if features.size > 0]
background_train_features = [features for features in background_train_features if features.size > 0]

foreground_train_labels = np.array([np.ones(features.shape[0]) for features in foreground_train_features])
background_train_labels = np.array([np.zeros(features.shape[0]) for features in background_train_features])

foreground_train_features = np.array(foreground_train_features)
background_train_features = np.array(background_train_features)

foreground_train_features_flattened = foreground_train_features.reshape(-1, foreground_train_features.shape[2])
background_train_features_flattened = background_train_features.reshape(-1, background_train_features.shape[2])

foreground_train_labels_flattened = np.concatenate(foreground_train_labels)
background_train_labels_flattened = np.concatenate(background_train_labels)

train_features = np.concatenate((foreground_train_features_flattened, background_train_features_flattened))
train_labels = np.concatenate((foreground_train_labels_flattened, background_train_labels_flattened))


train_features, train_labels = shuffle(train_features, train_labels, random_state=42)

print("Classifiers training started\n")
    
print("Training MLP classifier...")
train_mlp_classifier(train_features, train_labels)

print("Training SVM classifier...")
train_svm_classifier(train_features, train_labels)

print("Training LS classifier...")
train_lstsq_classifier(train_features, train_labels)

print("Training RNN classifier...")
train_features_rnn = train_features.reshape((-1, expected_frames, n_mels))
train_labels_rnn = train_labels.reshape((-1, expected_frames, 1))
train_rnn_classifier(train_features_rnn, train_labels_rnn)

