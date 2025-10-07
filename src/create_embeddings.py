from florence_pytorch.florence.configuration_florence2 import *
from florence_pytorch.florence.florence_attn import *
import florence_pytorch.florence.modeling_florence2 as flor2
from florence_pytorch.florence.processor import *
from PIL import Image
import csv
import pandas as pd
import torch
import json
import argparse
from .helper import get_flo_embeddings

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_embeddings(flo_model, processor, detailed_caption_file):
    df = pd.read_csv(detailed_caption_file, header=None, names=['label', 'definition'], engine='python')

    embedding_data = {}
    for index, row in df.iterrows():
        if index == 0:
            continue
        
        # Process label
        label = row['label']
        label_embedding = get_flo_embeddings(flo_model, processor, label)
        
        # Process definition
        definition = row['definition']
        def_embedding = get_flo_embeddings(flo_model, processor, definition)

        embedding_data[label] = {
            'label_embedding': label_embedding.detach().cpu(),
            'definition': definition,
            'def_embedding': def_embedding.detach().cpu()
        }
    
    return embedding_data
        
def create_data_files(embedding_data, output_dir):
    for label in embedding_data.keys():
        label_embedding = embedding_data[label]['label_embedding']
        definition = embedding_data[label]['definition']
        def_embedding = embedding_data[label]['def_embedding']

        # Save files
        label_save_path = os.path.join(os.path.join(output_dir, "labels"), f"{label.replace('/', '_').replace(' ', '_').replace('-', '_')}.pt")
        torch.save({
            'label': label,
            'embedding': label_embedding
        }, label_save_path)

        def_save_path = os.path.join(os.path.join(output_dir, "definition"), f"{label.replace('/', '_').replace(' ', '_').replace('-', '_')}_definition.pt")
        embedding_data[label] = {}
        torch.save({
            'label': label,
            'definition': definition,
            'embedding': def_embedding
        }, def_save_path)

        print(f'Processed label: {label}')
        print(f'Label embedding shape: {label_embedding.shape}, meaning {label_embedding.shape[0]-2} tokens plus start/end tokens')
        print(f'Processed definition for: {label}')
        print(f'Definition embedding shape: {def_embedding.shape}, meaning {def_embedding.shape[0]-2} tokens plus start/end tokens')
        print('-' * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--csv", default='./csv/label_definitions.csv', help="The name of the user to greet.")
    parser.add_argument("--output_dir", default='./tmp/', help="The name of the user to greet.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    
    flo_model, processor = flor2.load("BASE_FT", device)

    embedding_data = create_embeddings(flo_model, processor, args.csv)
    create_data_files(embedding_data, args.output_dir)