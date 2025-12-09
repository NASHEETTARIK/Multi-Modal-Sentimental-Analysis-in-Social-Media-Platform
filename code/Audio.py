import os
import re
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import butter, lfilter
import scipy.ndimage
import soundfile as sf
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
warnings.filterwarnings("ignore", message=".*build().*was called.*", module="keras")
warnings.filterwarnings("ignore", message=".*softmax over axis.*")
warnings.filterwarnings("ignore", message=".*Feedback manager requires a model with a single signature inference.*")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense

def Audio_Features() :
        
        
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]
    
    
    
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]
    
    
    root_dir = 'enterface database'
    
    file_paths = []
    
    # Iterate over all the directories and files in the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.wav'):
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)
    
    # Sort the file paths naturally
    file_paths = sorted(file_paths, key=natural_sort_key)
    
    """========================================================================
               The audio files are pre-processed using 
             Pass(low pass-Butterworth) Gaussian filter 
       ========================================================================""" 
    
    # Define Butterworth filter parameters
    cutoff = 3000.0  # Cutoff frequency in Hz
    order = 6        # Order of the filter
    
    # Define Gaussian filter parameters
    sigma = 1.0      # Standard deviation for Gaussian kernel
    
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def extract_features(file_path):
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        print(file_path)
        # Apply Butterworth low-pass filter
        y_filtered = butter_lowpass_filter(y, cutoff, sr, order)
        
        # Apply Gaussian filter
        y_smoothed = scipy.ndimage.gaussian_filter1d(y_filtered, sigma=sigma)
        
        return y_smoothed
    
    
    # Extract features, normalize, and append
    audio_f = []
    
    for file_path in file_paths:
        print(file_path)
        features = extract_features(file_path)
        audio_f.append(features)
    
    
    target_length = min(len(features) for features in audio_f)
    
    feature =[]
    
    for i in range(len(audio_f)):
        
        # Crop or pad the feature to match the target length
        if len(audio_f[i]) > target_length:
            # Crop if longer than target length
            y_smoothed = audio_f[i][:target_length]
            feature.append(y_smoothed)
        else:
            # Pad with zeros if shorter than target length
            y_smoothed = np.pad(audio_f[i], (0, target_length - len(audio_f[i])), 'constant')
            feature.append(y_smoothed)
        
    
    audio_f_np = np.array(feature)
    
    
    def create_cnn_feature_extractor(audio_f_np):
        # Define a simple 1D CNN model
        input_shape = (audio_f_np.shape[1], 1)  # (num_features, num_channels)
        inputs = Input(shape=input_shape)
        # Convolutional layers
        x = Conv1D(filters=32, kernel_size=3, activation='tanh')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, activation='tanh')(x)
        x = MaxPooling1D(pool_size=2)(x)
        # Flatten the output and pass to a dense layer
        x = Flatten()(x)
        x = Dense(1024, activation='tanh')(x)  # This layer acts as feature representation    
        # Define the model
        model = Model(inputs=inputs, outputs=x)
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
        return model
        
    
    
    feature_extractor =create_cnn_feature_extractor(audio_f_np)
    # feature_extractor.save('Features1/model_enterface database.h5')
    # Use the feature extractor to get features from your data
    features = feature_extractor.predict(audio_f_np)
    

    return features




















