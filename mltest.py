import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os

# Speech-to-Text (STT) Model
# Speech-to-Text (STT) Model
def build_stt_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(128)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Text-to-Speech (TTS) Model
def build_tts_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(128, return_sequences=True)(x)
    outputs = TimeDistributed(Dense(num_classes, activation='softmax'))(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Data Preparation (STT)
def prepare_data_stt(audio_file, target_text, num_classes):
    # Load audio file
    audio, sr = librosa.load(audio_file, sr=None)
    # Extract features (e.g., Mel spectrograms)
    features = librosa.feature.melspectrogram(audio, sr=sr, n_fft=400, hop_length=160, n_mels=num_classes)
    features = np.expand_dims(features, axis=-1)
    # Convert text to one-hot encoding
    target_text_onehot = tf.keras.utils.to_categorical(target_text, num_classes=num_classes)
    return features, target_text_onehot

# Data Preparation (TTS)
def prepare_data_tts(text_file, audio_file, num_classes):
    # Load text file and convert to one-hot encoding
    with open(text_file, 'r') as f:
        text = f.read()
    text_onehot = tf.keras.utils.to_categorical(text, num_classes=num_classes)
    # Load audio file and extract features
    audio, sr = librosa.load(audio_file, sr=None)
    features = librosa.feature.melspectrogram(audio, sr=sr, n_fft=400, hop_length=160, n_mels=num_classes)
    features = np.expand_dims(features, axis=-1)
    return text_onehot, features

# Example Usage
num_classes_stt = 28  # Example: 26 letters + space + blank
num_classes_tts = 128  # Example: ASCII characters
input_shape_stt = (None, 128, 1)  # Example: Mel spectrogram shape
input_shape_tts = (None, 128, 1)  # Example: Mel spectrogram shape

# Build and train STT model
stt_model = build_stt_model(input_shape_stt, num_classes_stt)
audio_file = 'sample_audio.wav'
target_text = np.array([0, 1, 2, ..., 26])  # Example: Convert transcript to numerical representation
features, target_text_onehot = prepare_data_stt(audio_file, target_text, num_classes_stt)
stt_model.fit(features, target_text_onehot, epochs=10, batch_size=32)

# Build and train TTS model
tts_model = build_tts_model(input_shape_tts, num_classes_tts)
text_file = 'sample_text.txt'
audio_file = 'sample_audio.wav'
text_onehot, features = prepare_data_tts(text_file, audio_file, num_classes_tts)
tts_model.fit(text_onehot, features, epochs=10, batch_size=32)

# Save models
stt_model.save('stt_model.h5')
tts_model.save('tts_model.h5')

# Inference (STT)
def predict_stt(audio_file, stt_model):
    audio, sr = librosa.load(audio_file, sr=None)
    features = librosa.feature.melspectrogram(audio, sr=sr, n_fft=400, hop_length=160, n_mels=num_classes_stt)
    features = np.expand_dims(features, axis=-1)
    prediction = stt_model.predict(features)
    predicted_text = np.argmax(prediction, axis=-1)
    return predicted_text

# Inference (TTS)
def predict_tts(text, tts_model):
    text_onehot = tf.keras.utils.to_categorical(text, num_classes=num_classes_tts)
    prediction = tts_model.predict(text_onehot)
    return prediction

# Example inference
predicted_text = predict_stt('sample_audio.wav', stt_model)
print(predicted_text)
predicted_audio = predict_tts('Hello, world!', tts_model)
print(predicted_audio)
