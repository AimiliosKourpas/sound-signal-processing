import os
import numpy as np
import json
import librosa
import joblib
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

test_audio_files = ['test1.wav', 'test2.wav', 'test3.wav']

n_fft = 1024
hop_length = 512
n_mels = 128
window_type = 'hann'
sr = 16000
tolerance = 0.0
expected_frames = 63
autocorr_threshold = 0.7 

def compute_mel_spectrogram(y, sr, n_fft, hop_length, n_mels, window_type):
    if len(y) == 0:
        return np.array([])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window=window_type)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.T

def process_audio_file(audio_path, sr):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if len(y) == 0:
            return np.array([])
        log_mel_spectrogram = compute_mel_spectrogram(y, sr, n_fft, hop_length, n_mels, window_type)
        return log_mel_spectrogram, y
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return np.array([]), None

def detect_conversations(sequence, threshold=0.5):
    conversations = []
    i = 0
    while i < len(sequence):
        while i < len(sequence) and sequence[i] < threshold:
            i += 1
        if i < len(sequence) and sequence[i] >= threshold:
            start = i
            while i < len(sequence) and sequence[i] >= threshold:
                i += 1
            end = i
            conversations.append((start, end))
        i += 1
    return conversations

def extract_actual_intervals(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    word_start_times = data["alignment"]["word_start_times_seconds"]
    word_end_times = data["alignment"]["word_end_times_seconds"]
    actual_intervals = list(zip(word_start_times, word_end_times))
    return actual_intervals

def read_words_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        words = file.read().split()
    return words

def detect_words(actual_intervals, predicted_intervals, words, tolerance):
    detected_words = []
    detected_intervals = []
    for word, (start, end) in zip(words, actual_intervals):
        for pred_start, pred_end in predicted_intervals:
            if interval_within_tolerance(start, end, [(pred_start, pred_end)], tolerance):
                detected_words.append(word)
                detected_intervals.append((pred_start, pred_end))
                break
    return detected_words, detected_intervals

def interval_within_tolerance(act_start, act_end, pred_intervals, tolerance):
    for pred_start, pred_end in pred_intervals:
        if (pred_start >= act_start - tolerance and pred_start <= act_end + tolerance) or \
           (pred_end >= act_start - tolerance and pred_end <= act_end + tolerance) or \
           (act_start >= pred_start - tolerance and act_end <= pred_end + tolerance):
            return True
    return False

def pad_spectrogram(spectrogram, expected_frames):
    if spectrogram.shape[0] > expected_frames:
        return spectrogram[:expected_frames]
    else:
        padding = np.zeros((expected_frames - spectrogram.shape[0], spectrogram.shape[1]))
        return np.vstack((spectrogram, padding))

def predict_rnn(rnn_model, log_mel_spectrogram, expected_frames, n_mels):
    num_segments = int(np.ceil(log_mel_spectrogram.shape[0] / expected_frames))
    padded_spectrogram = pad_spectrogram(log_mel_spectrogram, num_segments * expected_frames)
    reshaped_spectrogram = padded_spectrogram.reshape((num_segments, expected_frames, n_mels))
    predictions = rnn_model.predict(reshaped_spectrogram)
    return predictions.flatten()

def median_filter(sequence, kernel_size=5):
    assert kernel_size % 2 == 1, "Kernel size must be an odd integer"
    half_size = kernel_size // 2
    padded_sequence = np.pad(sequence, (half_size, half_size), mode="edge")
    filtered_sequence = np.zeros_like(sequence)
    for i in range(len(sequence)):
        filtered_sequence[i] = np.median(padded_sequence[i : i + kernel_size])
    return filtered_sequence

def compute_autocorrelation(signal, sr, threshold=0.7):
    if len(signal) == 0:
        return 0

    autocorr = librosa.autocorrelate(signal)
    autocorr /= np.max(autocorr)

    peaks = np.where(autocorr > threshold)[0]
    if len(peaks) == 0:
        return 0

    for peak in peaks:
        if peak > 0:
            f0 = sr / peak
            if 50 <= f0 <= 500:
                return f0
    return 0

def print_menu():
    print("Choose an audio file to test:")
    for i, file in enumerate(test_audio_files):
        print(f"{i + 1}. {file}")
    print("4. Process all files")
    while True:
        try:
            file_choice = int(input("Enter the number of the file: ")) - 1
            if 0 <= file_choice < len(test_audio_files) or file_choice == 3:
                break
            else:
                print("Enter a number between 1 and 4.")
        except ValueError:
            print("Enter a number between 1 and 4.")
    print("\nChoose a classifier to use:")
    classifiers = ['MLP', 'SVM', 'Least Squares', 'RNN']
    for i, clf in enumerate(classifiers):
        print(f"{i + 1}. {clf}")
    print("5. Use all classifiers")
    while True:
        try:
            classifier_choice = int(input("Enter the number of the classifier: ")) - 1
            if 0 <= classifier_choice < len(classifiers) or classifier_choice == 4:
                break
            else:
                print("Enter a number between 1 and 5.")
        except ValueError:
            print("Enter a number between 1 and 5.")
    return file_choice, classifier_choice

def process_and_evaluate(file_choice, classifier_choice):
    if file_choice == 3:
        audio_files = test_audio_files
    else:
        audio_files = [test_audio_files[file_choice]]
    classifiers = ['MLP', 'SVM', 'Least Squares', 'RNN']
    if classifier_choice == 4:
        selected_classifiers = classifiers
    else:
        selected_classifiers = [classifiers[classifier_choice]]
    mlp_model = load_model('mlp_classifier.h5')
    svm_model = joblib.load('svm_classifier.pkl')
    lstsq_weights = joblib.load('lstsq_classifier.pkl')
    rnn_model = load_model('rnn_classifier.h5')
    results = {}
    for audio_path in audio_files:
        print(f"\nProcessing {audio_path}...")
        log_mel_spectrogram, y = process_audio_file(audio_path, sr)
        if log_mel_spectrogram.size == 0:
            print(f"Warning: {audio_path} is empty or could not be loaded.")
            continue
        json_file = audio_path.replace('.wav', '.json')
        txt_file = audio_path.replace('.wav', '.txt')
        actual_intervals = extract_actual_intervals(json_file)
        words = read_words_from_txt(txt_file)
        print(f"Log Mel Spectrogram shape: {log_mel_spectrogram.shape}")
        for classifier in selected_classifiers:
            print(f"\nUsing {classifier} classifier...")
            if classifier == 'MLP':
                pred_prob = mlp_model.predict(log_mel_spectrogram)
            elif classifier == 'SVM':
                pred_prob = svm_model.decision_function(log_mel_spectrogram)
            elif classifier == 'Least Squares':
                log_mel_spectrogram_bias = np.hstack([np.ones((log_mel_spectrogram.shape[0], 1)), log_mel_spectrogram])
                pred_prob = log_mel_spectrogram_bias @ lstsq_weights
            elif classifier == 'RNN':
                pred_prob = predict_rnn(rnn_model, log_mel_spectrogram, expected_frames, n_mels)
            else:
                raise ValueError("Invalid classifier choice")
            pred = (pred_prob >= 0.5).astype(int).flatten()
            pred = median_filter(pred)
            conversations = detect_conversations(pred)
            print(f"{classifier} predictions: {pred.tolist()}")
            conversations_in_time = [(librosa.frames_to_time(start, sr=sr, hop_length=hop_length),
                                      librosa.frames_to_time(end, sr=sr, hop_length=hop_length))
                                     for start, end in conversations]
            detected_words, detected_intervals = detect_words(actual_intervals, conversations_in_time, words, tolerance)
            if audio_path not in results:
                results[audio_path] = {}
            f0_values = []
            for (start, end) in detected_intervals:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                word_signal = y[start_sample:end_sample]
                f0 = compute_autocorrelation(word_signal, sr)
                if f0 > 0:
                    f0_values.append(f0)
            avg_f0 = np.mean(f0_values) if f0_values else 0
            results[audio_path][classifier] = {
                'word_count': len(detected_words),
                'detected_words': list(zip(detected_words, detected_intervals)),
                'accuracy': accuracy_score((pred_prob >= 0.5).astype(int), np.array([1 if any(start <= t <= end for start, end in actual_intervals) else 0 for t in range(len(pred_prob))])),
                'average_f0': avg_f0
            }
    for audio_path, metrics in results.items():
        print(f"\nResults for {audio_path}:")
        for classifier, data in metrics.items():
            print(f"Classifier: {classifier}")
            print(f"Words Detected: {data['word_count']}")
            print(f"Accuracy: {data['accuracy']:.4f}")
            print(f"Average F0: {data['average_f0']:.2f} Hz")
            print("Detected Words and Intervals:")
            for word, interval in data['detected_words']:
                print(f"{interval}")

def normalize_features(features):
    return (features - np.mean(features)) / np.std(features)

if __name__ == "__main__":
    file_choice, classifier_choice = print_menu()
    process_and_evaluate(file_choice, classifier_choice)
