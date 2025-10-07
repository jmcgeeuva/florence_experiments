from .similarities import cross_similarity
from florence_pytorch.next_word import next_word_distribution
import torch
import os
import numpy as np
import pandas as pd
from glob import glob
import re
import torch.nn.functional as F
import argparse

def next_word_similarity(sentence1, sentence2):
    p1 = next_word_distribution(sentence1)
    p2 = next_word_distribution(sentence2)
    return F.cosine_similarity(p1, p2, dim=0)

def get_labels(label_file_path, target_labels):
    """Returns the label embeddings as a dictionary containing all definitions and embeddings for each label

    Parameters
    ----------
    label_file_path : string
        The path to the label file directory of embeddings (in .pt format)

    target_labels : list
        A list of strings that describe the labels in question
    """
    # Load label definition embeddings
    label_embeddings = {}
    label_files = glob(label_file_path)
    
    for label_file in label_files:
        data = torch.load(label_file)
        label = data['label']
        
        # Only process labels in our target list
        if label in target_labels:
            label_embeddings[label] = {
                'definition': data['definition'],
                'embedding': data['embedding']
            }
            print(f"Loaded label: {label}")
    return label_embeddings

def get_captions(caption_file_path, target_labels):
    """Returns all the caption embeddings and their coinciding caption text for each frame

    Parameters
    ----------
    caption_file_path : string
        The path to the caption file directory of caption definitions (in .pt format)

    target_labels : list
        A list of strings that describe the labels in question
    """
    # Load caption embeddings
    caption_embeddings = {}
    caption_files = glob(caption_file_path)
    
    for caption_file in caption_files:
        data = torch.load(caption_file)
        frame = os.path.basename(caption_file).replace('.pt', '')
        
        caption_embeddings[frame] = {
            'caption': data['caption'],
            'embedding': data['embedding']
        }
        print(f"Loaded caption for frame: {frame}")
    
    return caption_embeddings

def print_table_to_file(df_embedding, df_text):
    df_embedding.to_csv('cross_similarity_heatmap.csv')
    print(f"Saved embedding similarity matrix to 'cross_similarity_heatmap.csv'")
    
    df_text.to_csv('next_word_similarities_heatmap.csv')
    print(f"Saved text distribution similarity matrix to 'next_word_similarities_heatmap.csv'")

def print_results(df_embedding, df_text, caption_embeddings, axis_labels):

    print("\nEmbedding Similarity Statistics:")
    print(f"Average similarity: {df_embedding.values.mean():.4f}")
    print(f"Maximum similarity: {df_embedding.values.max():.4f}")
    print(f"Minimum similarity: {df_embedding.values.min():.4f}")
    
    print("\nText Distribution Similarity Statistics:")
    print(f"Average similarity: {df_text.astype(float).values.mean():.4f}")
    print(f"Maximum similarity: {df_text.astype(float).values.max():.4f}")
    print(f"Minimum similarity: {df_text.astype(float).values.min():.4f}")

    # Find top matches for each label using embedding similarity
    print("\nTop caption matches for each label (embedding similarity):")
    for label in axis_labels:
        top_matches = df_embedding.loc[label].astype(float).nlargest(3)
        print(f"\n{label}:")
        for frame, score in top_matches.items():
            print(f"  Frame {frame}: {score:.4f} - {caption_embeddings[frame]['caption'][:100]}...")
    
    # Find top matches for each label using text distribution similarity
    print("\nTop caption matches for each label (text distribution similarity):")
    for label in axis_labels:
        top_matches = df_text.loc[label].astype(float).nlargest(3)
        print(f"\n{label}:")
        for frame, score in top_matches.items():
            print(f"  Frame {frame}: {score:.4f} - {caption_embeddings[frame]['caption'][:100]}...")
            
def get_df(caption_embeddings, label_embeddings, target_labels, matrix):
    if type(list(caption_embeddings.keys())[0]) == str:
        axis_frames = sorted(caption_embeddings.keys(), key=lambda x: int(x) if x.isdigit() else x)
    else:
        axis_frames = sorted(caption_embeddings.keys())
    axis_labels = [label for label in target_labels if label in label_embeddings]
    df = pd.DataFrame(index=axis_labels, columns=axis_frames)
    for label in axis_labels:
        for frame in axis_frames:
            if frame in matrix[label]:
                df.loc[label, frame] = matrix[label][frame]
    return df

def create_table(caption_embeddings, target_labels, label_embeddings, embedding_similarity_matrix, text_similarity_matrix):
    # Convert to pandas DataFrame for easier analysis and saving
    
    # Initialize the DataFrame with NaN values
    

    # Fill in the values
    matrix = embedding_similarity_matrix
    df_embedding = get_df(caption_embeddings, label_embeddings, target_labels, matrix)
    matrix = text_similarity_matrix
    df_text = get_df(caption_embeddings, label_embeddings, target_labels, matrix)

    return df_embedding, df_text, axis_labels

# Calculate similarities
def calculate_similarity(label_embeddings, caption_embeddings, similarity_function, category, label_level, normalize=False):
    similarity_matrix = {}
    for label, embeddings in label_embeddings.items():
        label_element = embeddings[label_level]
        
        similarities = {}
        for frame, caption_data in caption_embeddings.items():
            caption_element = caption_data[category]
            if type(label_element) != str:
                label_element = label_element.to(caption_element.device)
            
            emb_similarity = similarity_function(label_element, caption_element)
            similarities[frame] = emb_similarity
            
        similarity_matrix[label] = similarities
        print(f"Computed similarities for label: {label}")

    if normalize:
        # Ensure consistent key ordering
        outer_keys = sorted(similarity_matrix.keys())
        inner_keys = sorted(next(iter(similarity_matrix.values())).keys())

        tensor = torch.tensor([[similarity_matrix[o][i] for i in inner_keys] for o in outer_keys])
        # tensor = tensor.permute(1, 0)
        tensor = torch.nn.functional.softmax(tensor, dim=0)
        # tensor = tensor.permute(1, 0)
        tmp_dict = {}
        for i, t in enumerate(tensor):
            key = outer_keys[i]
            tmp_dict[key] = {}
            for j, frame in enumerate(inner_keys):
                tmp_dict[key][frame] = t[j]
        
        return tmp_dict
    return similarity_matrix

def calculate_embedding_similarity(label_embeddings, caption_embeddings):
    similarity_matrix = calculate_similarity(label_embeddings, caption_embeddings, cross_similarity, category='embedding', label_level='embedding')
    return similarity_matrix

def calculate_text_similarity(label_embeddings, caption_embeddings):
    """Returns a similarity matrix for text caption similarities

    Parameters
    ----------
    label_embeddings : 

    caption_embeddings : 
    """
    similarity_matrix = calculate_similarity(label_embeddings, caption_embeddings, next_word_similarity, category='caption', label_level='definition')
    return similarity_matrix

def compute_cross_similarity(label_embeddings, caption_file_path, target_labels):
    """Runs the calculation for similarities of embeddings and texts

    Parameters
    ----------
    target_labels : list
        A list of strings that describe the labels in question

    Returns
    -------
    Nothing but instead creates a csv file at two locations one for text and the other for embeddings
    """
    embedding_similarity_matrix = calculate_embedding_similarity(label_embeddings, caption_embeddings)
    text_similarity_matrix = calculate_text_similarity(label_embeddings, caption_embeddings)
        
    df_embedding, df_text, axis_labels = create_table(caption_embeddings, target_labels, label_embeddings, embedding_similarity_matrix, text_similarity_matrix)
    return df_embedding, df_text, caption_embeddings, axis_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--eaf", default="./embeddings/eaf_definition_embeddings/*.pt", help="The name of the user to greet.")
    parser.add_argument("--caption", default="./embeddings/detailed_caption_embeddings/*.pt", help="The name of the user to greet.")
    args = parser.parse_args()

    # Define the labels of interest
    target_labels = ['Individual activity', 'Student writing', 'Individual technology', 'Sitting at desks', 'Student(s) standing or walking', 'Small group activity', 'Student raising hand']

    label_embeddings = get_labels(args.eaf, target_labels)
    caption_embeddings = get_captions(args.caption, target_labels)
    df_embedding, df_text, caption_embeddings, axis_labels = compute_cross_similarity(label_embeddings, caption_embeddings, target_labels)
    print_table_to_file(df_embedding, df_text)
    print_results(df_embedding, df_text, caption_embeddings, axis_labels)
