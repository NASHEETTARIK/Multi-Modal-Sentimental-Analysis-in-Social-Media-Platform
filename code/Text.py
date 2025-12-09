import os
import re
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
warnings.filterwarnings("ignore", message=".*build().*was called.*", module="keras")
warnings.filterwarnings("ignore", message=".*softmax over axis.*")
warnings.filterwarnings("ignore", message=".*Feedback manager requires a model with a single signature inference.*")
    

def Text_Features():
    
    """========================================================================
                  pre-processed using Tokenization and Stemming 
       ========================================================================"""    

    # Ensure you have NLTK data downloaded
    nltk.download('punkt')
    
    # # Load spaCy model
    # nlp = spacy.load('en_core_web_sm')
    
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load('en_core_web_sm')
        
    
    # Load your DataFrame
    df = pd.read_csv('enterface database/output.csv')
    
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
    
    """========================================================================
                       The features are extracted using
       Improved Term Frequency-Inverse Document Frequency (ITF-IDF) models
       ========================================================================"""    

    # the word’s length, the position of words and word’s speech parameter
    # Compute TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_content'])
    # joblib.dump(tfidf_vectorizer, f'Features/tfidf_vectorizer.pkl')
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    columns=tfidf_vectorizer.get_feature_names_out()
    # columns = np.save('Features/columns.npy',columns)
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
    itf_idf_df = pd.DataFrame(itf_idf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    
    return itf_idf_df







# dfff = Text_Features()



# def natural_sort_key(s):
#     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

# # Define the root directory
# root_dir = 'enterface database'

# # List to store file paths
# file_paths = []

# # Iterate over all the directories and files in the root directory
# for dirpath, dirnames, filenames in os.walk(root_dir):
#     for filename in filenames:
#         if filename.endswith('.avi'):
#             file_path = os.path.join(dirpath, filename)
#             file_paths.append(file_path)

# # Sort the file paths naturally
# file_paths = sorted(file_paths, key=natural_sort_key)

# labels =[]
# for file_path in file_paths:
#     parts = re.split(r'[\\/]', file_path)
#     label = parts[2]
#     labels.append(label)




# import pandas as pd

# label_series = pd.Series(labels)
# label_encoded, uniques = pd.factorize(label_series)



# from sklearn.feature_selection import f_classif
# import numpy as np

# X = dfff                    # ITF-IDF features
# y = label_encoded           # numeric labels

# F_values, p_values = f_classif(X, y)



# f_ratio_df = pd.DataFrame({
#     "Feature": X.columns,
#     "F_Ratio": F_values,
#     "P_Value": p_values
# })



