import os
import re
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
warnings.filterwarnings("ignore", message=".*build().*was called.*", module="keras")
warnings.filterwarnings("ignore", message=".*softmax over axis.*")
warnings.filterwarnings("ignore", message=".*Feedback manager requires a model with a single signature inference.*")
    
def Video_Features() :   
    
    # Natural sort function
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]
    
    # Define the root directory
    root_dir = 'enterface database'
    
    # List to store file paths
    file_paths = []
    
    # Iterate over all the directories and files in the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.avi'):
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)
    
    # Sort the file paths naturally
    file_paths = sorted(file_paths, key=natural_sort_key)
    
        
    """========================================================================
           The video clips are extracted into Key frames using 
               “Spherical Interpolation based Q-learning (SIQ)”
       ========================================================================""" 
    
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
        
    def extract_keyframes(video_file, max_keyframes=5, min_keyframes=5):
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
        
        
    key_frames1 =[]
    

    
    # Process all video files to extract keyframes
    for file_path in file_paths:
        key_frame = extract_keyframes(file_path)
        key_frames1.append(key_frame)
        print(len(key_frame))
    
    
    """========================================================================
                The geometric feature calculation 
       ========================================================================""" 
    
    
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
    
    
    def pad_sequences(sequences, maxlen=None):
        """Pads sequences to the same length."""
        if not maxlen:
            maxlen = max(len(seq) for seq in sequences)
        padded = np.zeros((len(sequences), maxlen))
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded
    
    edge_features_all = []
    area_features_all = []
    
    for i in range(len(key_frames1)):
        edge_features_video = []
        area_features_video = []
        
        for j in range(1, len(key_frames1[i])):
            neutral_image = key_frames1[i][j-1]
            expressive_image = key_frames1[i][j]
            
            edge_features, area_features = Geometric_features(neutral_image, expressive_image)
            
            edge_features_video.append(edge_features)
            area_features_video.append(area_features)
            
        # Convert to numpy arrays after padding
        edge_features_all.append(pad_sequences(edge_features_video))
        area_features_all.append(pad_sequences(area_features_video))
    
    
    vedio_feature1 =[]
    for i in range(len(edge_features_all)):
        print(i)
        
        flattened_features = edge_features_all[i]
        flattened_features= flattened_features.flatten()
        vedio_feature1.append(flattened_features)
    
    
    max_length = max(len(seq) for seq in vedio_feature1)
    print(max_length)
    
    padded_features = []
    for features in vedio_feature1:
        padded_feature = np.pad(features, (0, max_length - len(features)), 'constant')
        padded_features.append(padded_feature)
    
    # Convert the list to a numpy array for easier handling
    padded_features1 = np.array(padded_features)
    
    
    vedio_feature2 =[]
    for i in range(len(area_features_all)):
        print(i)
        
        flattened_features = area_features_all[i]
        flattened_features= flattened_features.flatten()
        vedio_feature2.append(flattened_features)
    
    
    max_length = max(len(seq) for seq in vedio_feature2)
    
    padded_features = []
    for features in vedio_feature2:
        padded_feature = np.pad(features, (0, max_length - len(features)), 'constant')
        padded_features.append(padded_feature)
    
    # Convert the list to a numpy array for easier handling
    padded_features2 = np.array(padded_features)
    
    
    return  padded_features1 ,padded_features2





