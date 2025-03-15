import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.io import wavfile
import cv2
import os
import random

def load_fer_dataset(data_path):
    images = []
    labels = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (48, 48))
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_audio(audio_path):
    sample_rate, audio_data = wavfile.read(audio_path)
    audio_data = audio_data.astype(np.float32)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = tf.audio.encode_wav(audio_data, sample_rate)
    audio_data = tf.squeeze(audio_data, axis=-1)
    return audio_data

def build_image_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    return model

def build_audio_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    return model

def train_models(images, labels, audio_data):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    images = images.reshape((images.shape[0], 48, 48, 1))
    images = images.astype('float32') / 255.0
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    image_model = build_image_model((48, 48, 1))
    image_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    image_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    audio_data = np.array([preprocess_audio(audio_path) for audio_path in audio_data])
    audio_data = audio_data.reshape((audio_data.shape[0], audio_data.shape[1], 1))
    X_train_audio, X_test_audio, y_train_audio, y_test_audio = train_test_split(audio_data, labels, test_size=0.2, random_state=42)
    audio_model = build_audio_model((audio_data.shape[1], 1))
    audio_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    audio_model.fit(X_train_audio, y_train_audio, epochs=10, validation_data=(X_test_audio, y_test_audio))
    return image_model, audio_model

def fusion_predict(image_model, audio_model, image_data, audio_data, image_weight=0.6, audio_weight=0.4):
    image_probs = image_model.predict(image_data)
    audio_probs = audio_model.predict(audio_data)
    fused_probs = image_weight * image_probs + audio_weight * audio_probs
    return np.argmax(fused_probs, axis=1)

def main():
    data_path = "path_to_fer_dataset"
    audio_paths = ["path_to_audio1.wav", "path_to_audio2.wav", "path_to_audio3.wav"]
    images, labels = load_fer_dataset(data_path)
    image_model, audio_model = train_models(images, labels, audio_paths)
    test_image = np.random.rand(1, 48, 48, 1)
    test_audio = np.random.rand(1, 16000, 1)
    predicted_label = fusion_predict(image_model, audio_model, test_image, test_audio)
    print(f"Predicted Label: {predicted_label}")

if __name__ == "__main__":
    main()