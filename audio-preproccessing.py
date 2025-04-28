import os
import librosa

def load_audio_files(directory):
    audio_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)
                audio, sr = librosa.load(file_path, sr=16000)  # Load audio at 16kHz
                audio_files.append((audio, sr))
    return audio_files

# Example usage:
audio_directory = 'librispeech'  # Ensure this points to the directory containing .wav files
audio_files = load_audio_files(audio_directory)

# Print out the number of audio files loaded to verify
print(f'Loaded {len(audio_files)} audio files.')
