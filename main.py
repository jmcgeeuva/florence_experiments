from src.similarities import cross_similarity
from src.create_embeddings import create_embeddings
from src.visualize import visualize_cross_similarity
from src.computations import get_captions, next_word_similarity, calculate_similarity, get_df
from src.timestamp_captions import get_timestamp_captions, create_caption_embeddings
from eaf_labels import get_eaf_labels

import matplotlib.pyplot as plt
import florence_pytorch.florence.modeling_florence2 as flor2
import argparse
import torch
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", default="/standard/spencerNSF/NeuralNetworksProjectVideos/314hours", help="The name of the user to greet.")
    parser.add_argument("--video", default="110.006.2018_ELA2_Year2_20180205", help="The name of the user to greet.")
    parser.add_argument("--csv", default='./csv/label_definitions.csv', help="The name of the user to greet.")
    parser.add_argument("--prompt", default="<MORE_DETAILED_CAPTION>", help="The name of the user to greet.")
    parser.add_argument('--timestamps', nargs='+', default=[3000, 3500, 4000, 4500, 5000, 5500, 6000])
    args = parser.parse_args()

    video_path = os.path.join(args.data_dir, "Videos_314 Hours", args.video + '.mp4')
    annot_path = os.path.join(args.data_dir, "Video Annotations_314 Hours", args.video + '.eaf')

    task_prompt = args.prompt
    timestamps = args.timestamps
    csv = args.csv

    eaf_dict = get_eaf_labels(annot_path, args.timestamps)
    in_targets = eaf_dict

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    flo_model, processor = flor2.load("BASE_FT", device)

    label_embeddings = create_embeddings(flo_model, processor, csv)
    targets = list(label_embeddings.keys())

    timestamp_dict, images = get_timestamp_captions(flo_model, processor, video_path, timestamps, task_prompt)
    caption_embeddings = create_caption_embeddings(flo_model, processor, timestamp_dict)
    
    # Calculate similarity scores
    
    # Fill in the values
    embedding_similarity_matrix = calculate_similarity(label_embeddings, 
                                                                 caption_embeddings, 
                                                                 cross_similarity, 
                                                                 category='embedding', 
                                                                 label_level='def_embedding'
                                                                )
    df_embedding = get_df(caption_embeddings, label_embeddings, targets, embedding_similarity_matrix)
    
    text_similarity_matrix = calculate_similarity(label_embeddings, 
                                                  caption_embeddings, 
                                                  next_word_similarity, 
                                                  category='caption', 
                                                  label_level='definition'
                                                 )
    df_text = get_df(caption_embeddings, label_embeddings, targets, text_similarity_matrix)
    
    
    # Load the CSV
    visualize_cross_similarity(df_embedding.astype(float), "emb_similarity_heatmap.png", red_labels=in_targets)
    visualize_cross_similarity(df_text.astype(float), "text_similarity_heatmap.png", red_labels=in_targets)

    for frame, image in images.items():
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(f'{frame}.png')