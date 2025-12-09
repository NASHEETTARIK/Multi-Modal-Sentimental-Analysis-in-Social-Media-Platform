import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import numpy as np
import os
import re
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import butter, lfilter
import scipy.ndimage
import soundfile as sf
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from moviepy.editor import VideoFileClip
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import cv2
import mediapipe as mp
from scipy.spatial import Delaunay
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
warnings.filterwarnings("ignore", message=".*build().*was called.*", module="keras")
warnings.filterwarnings("ignore", message=".*softmax over axis.*")
warnings.filterwarnings("ignore", message=".*Feedback manager requires a model with a single signature inference.*")
import Performance 


def extract_video_features(video_path):

    # Function to load video frames
    def load_video_frames(file_path):
        video_clip = VideoFileClip(file_path)
        frames = [frame for frame in video_clip.iter_frames()]
        video_clip.close()
        return frames
    
    # Function to extract features from a frame
    def extract_features(frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        hist = normalize(hist.reshape(1, -1)).flatten()
        return hist
    
    # Function to normalize feature vectors to unit vectors
    def normalize_to_unit_vector(vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    
    # Spherical interpolation function (Slerp)
    def slerp(p0, p1, t):
        """Interpolate between points p0 and p1 with t in [0, 1]."""
        p0 = normalize_to_unit_vector(p0)
        p1 = normalize_to_unit_vector(p1)
        dot_product = np.clip(np.dot(p0, p1), -1.0, 1.0)
        omega = np.arccos(dot_product)
        sin_omega = np.sin(omega)
        if sin_omega == 0:
            return (1.0 - t) * p0 + t * p1  # Linear interpolation if angle is zero
        return (np.sin((1.0 - t) * omega) / sin_omega) * p0 + (np.sin(t * omega) / sin_omega) * p1
    
    # Simplified Q-learning class
    class QLearning:
        def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
            self.q_table = np.zeros((n_states, n_actions))
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.epsilon = epsilon
        
        def choose_action(self, state):
            # Always exploit: choose the best action based on current Q-table
            return np.argmax(self.q_table[state])
        
        def update(self, state, action, reward, next_state):
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
            td_error = td_target - self.q_table[state, action]
            self.q_table[state, action] += self.learning_rate * td_error
    
    def extract_keyframes(video_file, max_keyframes=10, min_keyframes=10):
        frames = load_video_frames(video_file)
        features = [extract_features(frame) for frame in frames]
        
        num_frames = len(features)
        similarity_matrix = np.zeros((num_frames, num_frames))
        
        # Calculate similarity matrix using Spherical Interpolation (Slerp)
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                interpolated = slerp(features[i], features[j], 0.5)
                similarity = 1 - cosine(normalize_to_unit_vector(features[i]), interpolated)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric matrix
        
        # Initialize Q-learning
        q_learning = QLearning(n_states=num_frames, n_actions=2)
        
        # Q-learning setup: each frame is a state, actions are keyframe selection
        key_frames = []
        keyframe_indices = set()  # To keep track of selected keyframe indices
        
        for i in range(num_frames):
            state = i
            action = q_learning.choose_action(state)
            
            # Ensure we have at least `min_keyframes`
            if len(key_frames) < min_keyframes:
                action = 1
            
            # Ensure we do not exceed `max_keyframes`
            if len(key_frames) >= max_keyframes:
                break
            
            # Reward is based on similarity; you may define it differently
            reward = np.mean(similarity_matrix[i]) if action == 1 else 0
            next_state = (i + 1) % num_frames  # Example next state
            q_learning.update(state, action, reward, next_state)
            
            if action == 1 and i not in keyframe_indices:
                key_frames.append(frames[i])
                keyframe_indices.add(i)
        
        # Ensure at least one keyframe is selected
        if not key_frames:
            key_frames.append(frames[0])
            
        resized_keyframes = [cv2.resize(frame, (256, 256)) for frame in key_frames]
            
        return resized_keyframes
    
    
    # print("Select an Input video from enterface database\n")
    # inpath_video = askopenfilename(initialdir='enterface database')
    # file_path = 'enterface database/subject 1/anger/sentence 1/s1_an_1.avi' 
    key_frames = extract_keyframes(video_path)
    
    
    # Initialize Mediapipe Face Mesh and Drawing utilities
    mp_face_mesh = mp.solutions.face_mesh
    
    def calculate_edge_length(p1, p2):
        """Calculate the Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def calculate_triangle_area(p1, p2, p3):
        """Calculate the area of a triangle given its vertices."""
        a = calculate_edge_length(p1, p2)
        b = calculate_edge_length(p2, p3)
        c = calculate_edge_length(p3, p1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        return area
    
    def geometric_features(neutral_landmarks, expressive_landmarks):
        """Calculate edge and area features from facial landmarks."""
        # Ensure landmarks are in the form of np.array for easier calculations
        neutral_landmarks = np.array(neutral_landmarks)
        expressive_landmarks = np.array(expressive_landmarks)
        
        tri = Delaunay(neutral_landmarks)
        
        edge_features = []
        area_features = []
        
        for simplex in tri.simplices:
            p1, p2, p3 = neutral_landmarks[simplex]
            p1_e, p2_e, p3_e = expressive_landmarks[simplex]
            
            # Calculate edge lengths for neutral and expressive faces
            edge_lengths_n = [
                calculate_edge_length(p1, p2), 
                calculate_edge_length(p2, p3), 
                calculate_edge_length(p3, p1)
            ]
            edge_lengths_e = [
                calculate_edge_length(p1_e, p2_e), 
                calculate_edge_length(p2_e, p3_e), 
                calculate_edge_length(p3_e, p1_e)
            ]
            
            # Calculate edge features
            edge_features.extend([en - ee for en, ee in zip(edge_lengths_n, edge_lengths_e)])
            
            # Calculate areas for neutral and expressive faces
            area_n = calculate_triangle_area(p1, p2, p3)
            area_e = calculate_triangle_area(p1_e, p2_e, p3_e)
            
            # Calculate area features
            area_features.append(area_n - area_e)
        
        return np.array(edge_features), np.array(area_features)
    
    def get_landmarks_from_results(results):
        """Extract landmarks from Mediapipe results."""
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            return [(lm.x, lm.y) for lm in landmarks.landmark]
    
    
    def Geometric_features(neutral_image, expressive_image):
        """Extract geometric features between neutral and expressive images."""
        # Convert the images to RGB
        neutral_rgb = cv2.cvtColor(neutral_image, cv2.COLOR_BGR2RGB)
        expressive_rgb = cv2.cvtColor(expressive_image, cv2.COLOR_BGR2RGB)
        
        # Initialize the Face Mesh using a context manager
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            # Process the images to detect landmarks
            neutral_results = face_mesh.process(neutral_rgb)
            expressive_results = face_mesh.process(expressive_rgb)
            
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            # If landmarks are found, draw them on the neutral image
            if neutral_results.multi_face_landmarks:
                for face_landmarks in neutral_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=neutral_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())

            # If landmarks are found, draw them on the expressive image
            if expressive_results.multi_face_landmarks:
                for face_landmarks in expressive_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=expressive_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())

        # Display the images with the face mesh landmarks
            cv2.imshow("Neutral Image with Landmarks", neutral_image)
            cv2.imshow("Expressive Image with Landmarks", expressive_image)
        
            # Wait for a key press and close the windows
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Extract landmarks
            neutral_landmarks = get_landmarks_from_results(neutral_results)
            expressive_landmarks = get_landmarks_from_results(expressive_results)
            
            # Initialize edge and area features as empty arrays
            edge_features = np.array([])
            area_features = np.array([])
            
            if neutral_landmarks and expressive_landmarks:
                # Calculate geometric features
                edge_features, area_features = geometric_features(neutral_landmarks, expressive_landmarks)
        
        return edge_features, area_features
    
    
    
    edge_features_video = []
    area_features_video = []
    
    for j in range(1, len(key_frames)):
        neutral_image = key_frames[j - 1]
        expressive_image = key_frames[j]
        
        edge_features, area_features = Geometric_features(neutral_image, expressive_image)
    
        
        edge_features_video.append(edge_features)
        area_features_video.append(area_features)
        
    
    flattened_features =edge_features_video[0]
    flattened_features= flattened_features.flatten()
    max_length =10860 
    padded_feature = np.pad(flattened_features, (0, max_length - len(flattened_features)), 'constant')
    padded_features1 = np.array(padded_feature)
    padded_features1 =padded_features1.reshape( 1,padded_features1.shape[0])
    selected_features = np.load('Features/video1_selected_features.npy')
    selected_columns_indices = np.argsort(selected_features)[:len(selected_features)]
    selected_video1_features = padded_features1[:, selected_columns_indices]
    print('video feature1 :',padded_features1.shape)
    print('video selected_feature1 :',selected_video1_features.shape)
    
    flattened_features =area_features_video[0]
    flattened_features= flattened_features.flatten()
    max_length =3620 
    padded_feature = np.pad(flattened_features, (0, max_length - len(flattened_features)), 'constant')
    padded_features2 = np.array(padded_feature)
    padded_features2 =padded_features2.reshape( 1,padded_features2.shape[0])
    selected_features = np.load('Features/video2_selected_features.npy')
    selected_columns_indices = np.argsort(selected_features)[:len(selected_features)]
    selected_video2_features = padded_features2[:, selected_columns_indices]
    print('video feature2 :',padded_features2.shape)
    print('video selected_feature2 :',selected_video2_features.shape)
    
    return selected_video1_features, selected_video2_features

def extract_audio_features(audio_path):
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
        # print(file_path)
        # Apply Butterworth low-pass filter
        y_filtered = butter_lowpass_filter(y, cutoff, sr, order)
        
        # Apply Gaussian filter
        y_smoothed = scipy.ndimage.gaussian_filter1d(y_filtered, sigma=sigma)
        
        return y_smoothed
    
    
    
    # print("Select an Input audio from enterface database \n")
    # inpath_audio = askopenfilename(initialdir=f'{inpath_video}')
    features = extract_features(audio_path)
    
    target_length = 49392
    
    
    # Crop or pad the feature to match the target length
    if len(features) > target_length:
        # Crop if longer than target length
        y_smoothed = features[:target_length]
    
    else:
        # Pad with zeros if shorter than target length
        y_smoothed = np.pad(features, (0, target_length - len(features)), 'constant')
        
    
    audio_f_np = np.array(y_smoothed)
    
    # feature_extractor = load_model('Features/model_enterface database.h5')
    # Use the feature extractor to get features from your data
    
    
    from tensorflow.keras.models import load_model
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
    feature_extractor=load_model('Features/model_enterface database.h5')    
    
    
    
    audio_f_np = np.expand_dims(audio_f_np, axis=-1)  
    audio_f_np = np.expand_dims(audio_f_np, axis=0) 
    
    features = feature_extractor.predict(audio_f_np)
    selected_features = np.load('Features/Audio_selected_features.npy','r')
    selected_columns_indices = np.argsort(selected_features)[:len(selected_features)]
    selected_Audio_features = features[:, selected_columns_indices]
    
    print('Audio_features :',features.shape)
    print('selected_Audio_features :',selected_Audio_features.shape)

    return selected_Audio_features

def extract_text_features(text_path):
    # Ensure you have NLTK data downloaded
    nltk.download('punkt')
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    
    with open(text_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
    
    # Create a DataFrame
    df = pd.DataFrame(content, columns=['content'])
    
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    
    # Function to tokenize and stem text
    def tokenize_and_stem(text):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    
    # Apply the function to your text column
    df['processed_content'] = df['content'].apply(tokenize_and_stem)
    
    from joblib import load
    
    tfidf_vectorizer = load('Features/tfidf_vectorizer.pkl')
    # Compute TF-IDF
    # tfidf_vectorizer = TfidfVectorizer()
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_content'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    # Analyze POS frequencies
    def pos_frequencies(text):
        doc = nlp(text)
        pos_counts = {}
        for token in doc:
            pos_tag = token.pos_
            if pos_tag not in pos_counts:
                pos_counts[pos_tag] = 0
            pos_counts[pos_tag] += 1
        return pos_counts
    
    # Aggregate POS frequencies for the entire dataset
    pos_counts = {}
    for text in df['content']:
        frequencies = pos_frequencies(text)
        for pos, count in frequencies.items():
            if pos not in pos_counts:
                pos_counts[pos] = 0
            pos_counts[pos] += count
    
    # Compute total counts and assign weights
    total_pos_count = sum(pos_counts.values())
    pos_weights = {pos: total_pos_count / count for pos, count in pos_counts.items()}
    
    # Function to compute POS factor based on dynamic weights
    def pos_factor(token):
        pos_tag = token.pos_
        return pos_weights.get(pos_tag, 1.0)
    
    # Compute factors for ITF-IDF
    def compute_factors(text):
        doc = nlp(text)
        factors = []
        for i, token in enumerate(doc):
            token_text = token.text
            length = len(token_text)
            position = 1 / (i + 1)  # Simple inverse position factor
            pos = pos_factor(token)
            factors.append(length * position * pos)
        return factors
    
    df['factors'] = df['processed_content'].apply(compute_factors)
    
    # Apply ITF-IDF
    def apply_itf_idf(tfidf_matrix, factors):
        itf_idf_matrix = tfidf_matrix.copy()
        for i, factor_list in enumerate(factors):
            for j, factor in enumerate(factor_list):
                if j < itf_idf_matrix.shape[1]:
                    itf_idf_matrix[i, j] *= factor
        return itf_idf_matrix
    
    itf_idf_matrix = apply_itf_idf(tfidf_matrix, df['factors'].tolist())
    
    columns1=tfidf_vectorizer.get_feature_names_out()
    columns = np.load('Features/columns.npy',allow_pickle=1)
    itf_idf_df = pd.DataFrame(itf_idf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    final_df = pd.DataFrame(0, index=itf_idf_df.index, columns=columns)
    
    # Fill final_df with values from itf_idf_df where column names match
    for col in columns1:
        if col in columns:
            final_df[col] = itf_idf_df[col]
    
    
    selected_features = np.load('Features/Text_selected_features.npy')
    
    selected_columns_indices = np.argsort(selected_features)[:len(selected_features)]
    selected_text_features = final_df.iloc[:, selected_columns_indices]

    print('text_features :',final_df.shape)
    print('selected_text_features :',selected_text_features.shape)
    return selected_text_features

def predict_text(text_features):

    from SA_CBiLSTM_Text_model import Test_Text   
    Text_pred = Test_Text(text_features)
    predicted = np.argmax(Text_pred, axis=-1)
    
    classes=["anger", "disgust" , "fear","happiness" ,"sadness","surprise"]

    return Text_pred,classes[predicted[0]]

def predict_audio(audio_features):
    from DCRNA_Audio_model import Test_Audio   
    Audio_pred = Test_Audio(audio_features)
    predicted = np.argmax(Audio_pred, axis=-1)
    
    classes=["anger", "disgust" , "fear","happiness" ,"sadness","surprise"]
    return Audio_pred ,classes[predicted[0]]

def predict_video(video_features1, video_features2):

    from GRCAT_Video_model import Test_Video   
    Video_pred = Test_Video(video_features1,video_features2)
    predicted = np.argmax(Video_pred, axis=-1)
    
    classes=["anger", "disgust" , "fear","happiness" ,"sadness","surprise"]
    return Video_pred,classes[predicted[0]]

def fuse_predictions(text_pred, audio_pred, video_pred, beta=0.6):

    def weighted_fusion(pv, pa,pc,beta):
        """
        Perform weighted fusion of visual and audio modality probabilities.
        Parameters:
            pv (numpy array): Posterior probabilities from the visual modality.
            pa (numpy array): Posterior probabilities from the audio modality.
            beta (float): Weight for the visual modality.
        Returns:
            numpy array: Fused posterior probabilities.
        """
        # Ensure the input arrays are numpy arrays
        pv = np.array(pv)
        pa = np.array(pa)
        pc = np.array(pc)
        # Compute the fused probabilities using the provided formula
        fused_probs = np.maximum(beta * pv, (1 - beta) * pa,(2 - beta) * pc)
        return fused_probs
    # Example usage
    # Posterior probabilities for 3 classes (example data)
    pv = np.array(text_pred)
    pa = np.array(audio_pred)
    pc = np.array(video_pred)

    # Weight for the visual modality (beta)
    beta = 0.6
    # Perform fusion
    fused_probs = weighted_fusion(pv, pa,pc, beta)
    predicted = np.argmax(fused_probs, axis=-1)
    
    classes=["anger", "disgust" , "fear","happiness" ,"sadness","surprise"]
    

    return classes[predicted[0]]


import tkinter as tk
from tkinter import filedialog, messagebox
import threading

class App:
    def __init__(self, master):
        self.master = master
        master.title("Enterface Database Processor")

        # Video File Selection
        self.video_label = tk.Label(master, text="Select Video File:")
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.video_entry = tk.Entry(master, width=50)
        self.video_entry.grid(row=0, column=1, padx=10, pady=10)

        # Audio File Selection
        self.audio_label = tk.Label(master, text="Select Audio File:")
        self.audio_label.grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.audio_entry = tk.Entry(master, width=50)
        self.audio_entry.grid(row=1, column=1, padx=10, pady=10)

        # Text File Selection
        self.text_label = tk.Label(master, text="Select Text File:")
        self.text_label.grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.text_entry = tk.Entry(master, width=50)
        self.text_entry.grid(row=2, column=1, padx=10, pady=10)

        # Browse Button for all files
        self.browse_button = tk.Button(master, text="Browse", command=self.browse_files)
        self.browse_button.grid(row=0, column=2, rowspan=3, padx=10, pady=10)

        # Beta Slider for Fusion
        self.beta_label = tk.Label(master, text="Fusion Weight (Beta):")
        self.beta_label.grid(row=3, column=0, padx=10, pady=10, sticky='e')
        self.beta_scale = tk.Scale(master, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.beta_scale.set(0.6)
        self.beta_scale.grid(row=3, column=1, padx=10, pady=10, sticky='w')

        # Process Button
        self.process_button = tk.Button(master, text="Process", command=self.process_files)
        self.process_button.grid(row=4, column=1, padx=10, pady=20)

        self.Perform = tk.Button(master, text="Performance", command=self.Plot)
        self.Perform.grid(row=4, column=2, padx=10, pady=20)

        # Exit Button
        self.exit_button = tk.Button(master, text="Exit", command=self.exit_application)
        self.exit_button.grid(row=4, column=3, padx=10, pady=20)

        # Results Display
        self.results_text = tk.Text(master, height=10, width=70)
        self.results_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

 
    def Plot(self):
        Performance.plot()


    
    def browse_files(self):
        # Open file dialog to select a video file
        video_filepath = filedialog.askopenfilename(
            initialdir='enterface database',
            title="Select Video File",
            filetypes=[("Video files", "*.avi")]
        )
    
        if video_filepath:  # Check if a video file was selected
            # Get the directory of the selected video file
            video_directory = os.path.dirname(video_filepath)
    
            # Open file dialog to select an audio file from the same directory
            audio_filepath = filedialog.askopenfilename(
                initialdir=video_directory,
                title="Select Audio File",
                filetypes=[("Audio files", "*.wav")]
            )
    
            if audio_filepath:  # Check if an audio file was selected
                # Open file dialog to select a text file from the same directory
                text_filepath = filedialog.askopenfilename(
                    initialdir=video_directory,
                    title="Select Text File",
                    filetypes=[("Text files", "*.txt")]
                )
    
                # Check if a text file was selected
                if text_filepath:
                    # Assign files to respective entry fields
                    self.video_entry.delete(0, tk.END)
                    self.video_entry.insert(0, video_filepath)
    
                    self.audio_entry.delete(0, tk.END)
                    self.audio_entry.insert(0, audio_filepath)
    
                    self.text_entry.delete(0, tk.END)
                    self.text_entry.insert(0, text_filepath)
                else:
                    messagebox.showwarning("Warning", "Please select a text file.")
            else:
                messagebox.showwarning("Warning", "Please select an audio file.")
        else:
            messagebox.showwarning("Warning", "Please select a video file.")


    def process_files(self):
        video_path = self.video_entry.get()
        audio_path = self.audio_entry.get()
        text_path = self.text_entry.get()
        beta = self.beta_scale.get()

        if not video_path or not audio_path or not text_path:
            messagebox.showerror("Error", "Please select all three files.")
            return

        # Disable the process button to prevent multiple clicks
        self.process_button.config(state=tk.DISABLED)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Processing...\n")

        # Run processing in a separate thread to keep the UI responsive
        threading.Thread(target=self.run_processing, args=(video_path, audio_path, text_path, beta)).start()

    def exit_application(self):
        # Confirm before closing the application
        if messagebox.askokcancel("Quit", "Do you really want to quit?"):
            self.master.destroy()

    def run_processing(self, video_path, audio_path, text_path, beta):
        try:
            # Extract Features
            self.results_text.insert(tk.END, "Extracting video features...\n")
            video_features1, video_features2 = extract_video_features(video_path)

            self.results_text.insert(tk.END, "Extracting audio features...\n")
            audio_features = extract_audio_features(audio_path)

            self.results_text.insert(tk.END, "Extracting text features...\n")
            text_features = extract_text_features(text_path)

            # Make Predictions
            self.results_text.insert(tk.END, "Predicting text...\n")
            text_pred,a = predict_text(text_features)
            self.results_text.insert(tk.END, f"Text Data sentimental analysis :{a}\n")
            
            self.results_text.insert(tk.END, "Predicting audio...\n")
            audio_pred,b = predict_audio(audio_features)
            self.results_text.insert(tk.END, f"Audio Data prediction sentimental analysis :{b}\n")

            self.results_text.insert(tk.END, "Predicting video...\n")
            video_pred,c = predict_video(video_features1, video_features2)
            self.results_text.insert(tk.END, f"Video Data prediction sentimental analysis :{c}\n")

            # Fuse Predictions
            self.results_text.insert(tk.END, "Fusing predictions...\n")
            fused_probs = fuse_predictions(text_pred, audio_pred, video_pred, beta)

            # Display Results
            self.results_text.insert(tk.END, "Processing completed.\n")
            self.results_text.insert(tk.END, f"prediction sentimental analysis :{fused_probs}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.results_text.insert(tk.END, f"Error: {e}\n")
        finally:
            # Re-enable the process button
            self.process_button.config(state=tk.NORMAL)

# Main Function to Run the GUI
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()