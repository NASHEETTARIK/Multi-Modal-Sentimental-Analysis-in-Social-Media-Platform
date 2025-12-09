import os
import re
from moviepy.editor import VideoFileClip

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

print(file_paths)

# Process each file
for file_path in file_paths:
    print(f"Processing file: {file_path}")
    try:
        # Load the video file
        video_clip = VideoFileClip(file_path)
        # Extract audio
        audio_clip = video_clip.audio
        
        # Define the audio output path
        audio_output_path = os.path.splitext(file_path)[0] + '.wav'
        
        # Write the audio file
        audio_clip.write_audiofile(audio_output_path)
        
        # Close the clips
        audio_clip.close()
        video_clip.close()
        
        print(f"Audio saved at: {audio_output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")



#%%
import assemblyai as aai
import os
import re
import speech_recognition as sr

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



for file_path in file_paths:
    # Transcribe the audio
    file_path =file_paths[0]
    aai.settings.api_key = "702404d00107482497bb621d35234753"
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(language_code='en')  # 'zh' for Chinese

    transcript = transcriber.transcribe(file_path,config=config)
    text = transcript.text
    print(text)
    text_output_path = os.path.splitext(file_path)[0] + '.txt'
    
    # Save transcription to a text file
    with open(text_output_path, 'w',encoding='utf-8') as f:
        f.write(f"{text}\n")
        # print(text)
    
    print(f"Processed {file_path}, saved transcription to {text_output_path}")    
    
    
    
    
#%%    
import csv    
    
root_dir = 'enterface database'  # Update this path to your dataset location

file_paths = []

# Function to naturally sort file paths
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# Iterate over all the directories and files in the root directory
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.txt'):
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)

# Sort the file paths naturally
file_paths = sorted(file_paths, key=natural_sort_key)


# Open a CSV file to write the data
csv_file_path = 'output.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['filename', 'content'])  # Writing the header

    for file_path in file_paths:
        # Open the file in read mode ('r')
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the entire content of the file
            content = file.read()
            # Get the filename without directory
            filename = os.path.basename(file_path)
            # rite filename and content to the CSV file
            csvwriter.writerow([filename, content])

print(f"Data saved to {csv_file_path}")        
    
    
    
    
#%%    
import assemblyai as aai
import os
import re
import speech_recognition as sr

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

root_dir = 'BAUM1s_MP4 - All'

file_paths = []

# Iterate over all the directories and files in the root directory
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.wav'):
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)

# Sort the file paths naturally
file_paths = sorted(file_paths, key=natural_sort_key)



# Transcribe audio and save to text files
for file_path in file_paths:
    # Transcribe the audio
    aai.settings.api_key = "702404d00107482497bb621d35234753"
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(language_code='en')  # 'zh' for Chinese

    transcript = transcriber.transcribe(file_path,config=config)
    text = transcript.text
    
    text_output_path = os.path.splitext(file_path)[0] + '.txt'
    
    # Save transcription to a text file
    with open(text_output_path, 'w',encoding='utf-8') as f:
        f.write(f"{text}\n")
        print(text)
    
    print(f"Processed {file_path}, saved transcription to {text_output_path}")    
    
    
#%%
import os
import re
from moviepy.editor import VideoFileClip

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

# Define the root directory
root_dir = 'BAUM1s_MP4 - All'

# List to store file paths
file_paths = []

# Iterate over all the directories and files in the root directory
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.mp4'):
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)

# Sort the file paths naturally
file_paths = sorted(file_paths, key=natural_sort_key)

print(file_paths)

# Process each file
for file_path in file_paths:
    print(f"Processing file: {file_path}")
    try:
        # Load the video file
        video_clip = VideoFileClip(file_path)
        # Extract audio
        audio_clip = video_clip.audio
        
        # Define the audio output path
        audio_output_path = os.path.splitext(file_path)[0] + '.wav'
        
        # Write the audio file
        audio_clip.write_audiofile(audio_output_path)
        
        # Close the clips
        audio_clip.close()
        video_clip.close()
        
        print(f"Audio saved at: {audio_output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")



# import jieba

# text = "我喜欢学习自然语言处理"
# tokens = jieba.lcut(text)
# print(tokens)



# import opencc

# converter = opencc.OpenCC('t2s')  # Convert Traditional to Simplified
# simplified_text = converter.convert("学习")
# print(simplified_text)



# #%%

# import jieba
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Sample Chinese documents
# documents = [
#     "我喜欢学习自然语言处理",  # I like studying natural language processing
#     "自然语言处理是人工智能的一个重要领域",  # NLP is an important field of AI
#     "我正在学习如何使用Python处理文本数据"  # I am learning how to use Python to process text data
# ]

# # Define a custom tokenizer using jieba
# def jieba_tokenizer(text):
#     return jieba.lcut(text)

# # Create a TF-IDF Vectorizer, specifying the tokenizer
# vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)

# # Fit and transform the documents
# tfidf_matrix = vectorizer.fit_transform(documents)

# # Get feature names (words)
# feature_names = vectorizer.get_feature_names_out()

# aaa =tfidf_matrix.toarray()

# # Print the TF-IDF matrix
# print("TF-IDF Matrix:")
# print(tfidf_matrix.toarray())

# # Print the feature names
# print("\nFeature Names:")
# print(feature_names)





#%%



# import os
# import re
# from moviepy.editor import VideoFileClip

# file_path = 'S001_001.mp4'

# video_clip = VideoFileClip(file_path)
# # Extract audio
# audio_clip = video_clip.audio

# audio_output_path ='S001_001.wav'

# audio_clip.write_audiofile(audio_output_path)
    
    