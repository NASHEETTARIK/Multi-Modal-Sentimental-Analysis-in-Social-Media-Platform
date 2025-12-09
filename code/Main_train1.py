import os
import re
import csv
import pandas as pd
import numpy as np
from Text import Text_Features
from Audio import Audio_Features
from Video import Video_Features



def Training() :

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
    
    labels =[]
    for file_path in file_paths:
        parts = re.split(r'[\\/]', file_path)
        label = parts[2]
        labels.append(label)
        
    
    from sklearn.preprocessing import OneHotEncoder
    labels_array = np.array(labels)
    labels_reshaped = labels_array.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(labels_reshaped)    
        
        
    text = Text_Features()
    
    """========================================================================
                  Adapted Firefly Optimization Algorithm  
       ========================================================================"""   
    
    
    from Text_Feature_selection import firefly_algorithms
    best_solution = firefly_algorithms(text,encoded_data)
    selected_features = np.where(best_solution == 1)[0]
    # np.save('Features/Text_selected_features.npy',selected_features)
    selected_features = np.load('Features/Text_selected_features.npy')

    selected_columns_indices = np.argsort(selected_features)[:len(selected_features)]
    selected_text_features = text.iloc[:, selected_columns_indices]
        
    audio = Audio_Features()
    
    """========================================================================
             Enhanced Genetic Grey lag goose optimization   
       ========================================================================"""  


    from Audio_Feature_selection import GGO
    dimensions = audio.shape[1]
    best_solution = GGO(dimensions,audio, encoded_data)
    selected_features = np.where(best_solution == 1)[0]
    # np.save('Features/Audio_selected_features.npy',selected_features)
    selected_features = np.load('Features/Audio_selected_features.npy')
    selected_columns_indices = np.argsort(selected_features)[:len(selected_features)]
    selected_Audio_features = audio[:, selected_columns_indices]


    
    video1,video2 = Video_Features()

    """========================================================================
                     Parrot optimization algorithm   
       ========================================================================"""  

    from video_Feature_selection import PO     
    best_solution = PO(video1, encoded_data)
    selected_features =  np.where(best_solution > 0.5)[0]
    # np.save('Features/video1_selected_features.npy',selected_features)
    selected_features = np.load('Features/video1_selected_features.npy')
    selected_columns_indices = np.argsort(selected_features)[:len(selected_features)]
    selected_video1_features = video1[:, selected_columns_indices]
    
    best_solution = PO(video2, encoded_data)
    selected_features =  np.where(best_solution > 0.5)[0]
    # np.save('Features/video2_selected_features.npy',selected_features)
    selected_features = np.load('Features/video2_selected_features.npy')
    selected_columns_indices = np.argsort(selected_features)[:len(selected_features)]
    selected_video2_features = video2[:, selected_columns_indices]
    
    """========================================================================
                        Data split to  train and test 
       ========================================================================"""  
       
    from sklearn.model_selection import train_test_split
        
    
    X_train_text, X_test_text, X_train_audio, X_test_audio, X_train_video1, X_test_video1, X_train_video2, X_test_video2, y_train, y_test = train_test_split(
        selected_text_features, selected_Audio_features, selected_video1_features, selected_video2_features, encoded_data, 
        test_size=0.2, random_state=42
    )

    # np.save('Features/X_train_text_enterface database.npy', X_train_text)
    # np.save('Features/X_test_text_enterface database.npy', X_test_text)
    
    # # Save training and testing sets for audio
    # np.save('Features/X_train_audio_enterface database.npy', X_train_audio)
    # np.save('Features/X_test_audio_enterface database.npy', X_test_audio)
    
    # # Save training and testing sets for video1
    # np.save('Features/X_train_video1_enterface database.npy', X_train_video1)
    # np.save('Features/X_test_video1_enterface database.npy', X_test_video1)
    
    # # Save training and testing sets for video2
    # np.save('Features/X_train_video2_enterface database.npy', X_train_video2)
    # np.save('Features/X_test_video2_enterface database.npy', X_test_video2)
    
    # # Save training and testing sets for encoded data
    # np.save('Features/y_train_enterface database.npy', y_train)
    # np.save('Features/y_test_enterface database.npy', y_test)    
    
    
        
    X_train_text = np.load('Features/X_train_text_enterface database.npy')
    # Save training and testing sets for audio
    X_train_audio = np.load('Features/X_train_audio_enterface database.npy')
    # Save training and testing sets for video1
    X_train_video1 = np.load('Features/X_train_video1_enterface database.npy')
    # Save training and testing sets for video2
    X_train_video2 = np.load('Features/X_train_video2_enterface database.npy')
    
    y_train =np.load('Features/y_train_enterface database.npy')
    y_test = np.load('Features/y_test_enterface database.npy')    
    

    """========================================================================
                  Self Attention based Capsule Bi-lstm  
       ========================================================================"""   
    from SA_CBiLSTM_Text_model import Train_Text   
    model_Text = Train_Text(X_train_text,y_train)    
    # model_Text.save('Features/model_Text.h5')
    
    """========================================================================
               Existing Models - Bidirectional GRU ,GRU ,CNN  
       ========================================================================"""  
       
    from Existing_text import Existing_Train_Text1  
    Existing_model_Text1 = Existing_Train_Text1(X_train_text,y_train)   
    # Existing_model_Text1.save('Features/GRU.h5')
    
    from Existing_text import Existing_Train_Text2  
    Existing_model_Text2 = Existing_Train_Text2(X_train_text,y_train)   
    # Existing_model_Text2.save('Features/BiGRU.h5')    
    
    from Existing_text import Existing_Train_Text3  
    Existing_model_Text3 = Existing_Train_Text3(X_train_text,y_train)   
    # Existing_model_Text3.save('Features/CNN.h5')    
    
    """========================================================================
           Gated attention enclosed Residual context aware transformer  
       ========================================================================"""
    
    
    from GRCAT_Video_model import Train_Video   
    model_Video = Train_Video(X_train_video1, X_train_video2,y_train)
    # model_Video.save('Features/model_Video.h5')
    
    """========================================================================
             Existing Models - CNN-BiLSTM ,BiLSTM ,LDCNN      
       ========================================================================"""  
    
    from Existing_video import Existing_Train_Video1   
    Existing_model_Video1 = Existing_Train_Video1(X_train_video1, X_train_video2,y_train)
    # Existing_model_Video1.save('Features/LDCNN.h5')
    
    from Existing_video import Existing_Train_Video2   
    Existing_model_Video2 = Existing_Train_Video2(X_train_video1, X_train_video2,y_train)
    # Existing_model_Video2.save('Features/CNN-BiLSTM.h5')
    
    from Existing_video import Existing_Train_Video3   
    Existing_model_Video3 = Existing_Train_Video3(X_train_video1, X_train_video2,y_train)
    # Existing_model_Video3.save('Features/BiLSTM.h5')
    
    """========================================================================
             Densely connected recurrent network with dual Attention    
       ========================================================================"""  
       
    from DCRNA_Audio_model import Train_Audio   
    model_Audio = Train_Audio( X_train_audio,y_train)  
    # model_Audio.save('Features/model_Audio.h5') 
       
    """========================================================================
             Existing Models - resnet50,VGG16 ,LSTM     
       ========================================================================"""  
    
    
    from Existing_audio import Existing_Train_Audio1   
    Existing_model_Audio1 = Existing_Train_Audio1( X_train_audio,y_train)  
    # Existing_model_Audio1.save('Features/resnet50.h5')
    
    from Existing_audio import Existing_Train_Audio2  
    Existing_model_Audio2 = Existing_Train_Audio2( X_train_audio,y_train)  
    # Existing_model_Audio2.save('Features/VGG16.h5')
    
    
    from Existing_audio import Existing_Train_Audio3   
    Existing_model_Audio3 = Existing_Train_Audio3( X_train_audio,y_train)  
    # Existing_model_Audio3.save('Features/LSTM.h5')
    

    
Training()    
    
    
    
    
    
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            