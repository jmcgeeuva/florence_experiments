from src.similarities import cross_similarity
from src.create_embeddings import create_embeddings as create_label_embeddings
from src.visualize import visualize_cross_similarity
from src.computations import get_captions, next_word_similarity, calculate_similarity, get_df
from src.timestamp_captions import get_timestamp_captions, create_caption_embeddings, get_images, extract_region_and_label_embeddings
from src.eaf_labels import get_eaf_labels

import matplotlib.pyplot as plt
import florence_pytorch.florence.modeling_florence2 as flor2
import argparse
import torch
import os

###### LISA ######
import sys, csv, json, cv2 
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor, GPT2Tokenizer, GPT2LMHeadModel
import xml.etree.ElementTree as ET
from collections import defaultdict
from glob import glob

# from florence.configuration_florence2 import *
# from florence.florence_attn import *
# import florence.modeling_florence2 as flor2
# from florence.processor import *

###### LISA ######

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--data_dir", default="/standard/spencerNSF/NeuralNetworksProjectVideos/314hours", help="The name of the user to greet.")
    # parser.add_argument("--video", default="110.006.2018_ELA2_Year2_20180205", help="The name of the user to greet.")
    parser.add_argument("--video", default="/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Videos_314 Hours/110.006.2018_ELA2_Year2_20180205.mp4", help="The name of the user to greet.")
    parser.add_argument("--eaf", default="/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Video Annotations_314 Hours/110.006.2018_ELA2_Year2_20180205.eaf", help="The name of the user to greet.")
    parser.add_argument("--csv", default='./csv/label_definitions.csv', help="The name of the user to greet.")
    parser.add_argument("--prompt", default="<MORE_DETAILED_CAPTION>", help="The name of the user to greet.")
    parser.add_argument('--timestamps', nargs='+') #, default=[3000, 3500, 4000, 4500, 5000, 5500, 6000])
    args = parser.parse_args()

    # video_path = os.path.join(args.data_dir, "Videos_314 Hours", args.video + '.mp4')
    # annot_path = os.path.join(args.data_dir, "Video Annotations_314 Hours", args.video + '.eaf')

    task_prompt = args.prompt
    csv = args.csv
    annot_path = args.eaf
    video_path = args.video


    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    print(f"Using device: {device}")
    flo_model, processor = flor2.load("BASE_FT", device, lora=False)
    flo_model = flo_model.to(device)
    # next_word_model = NextWordModel(device=device)

    # 1) Parse EAF
    frame_annotations = get_eaf_labels(annot_path, args.timestamps)

    # 2) Timestamps (if provided use them, otherwise derive from EAF keys)
    if args.timestamps:
        timestamps = args.timestamps
    else:
        timestamps = sorted(frame_annotations.keys())[::(1800*30)][:6]  # default to first 100 if many
    print(f"Processing timestamps: {timestamps}")

    # 3) For each timestamp: extract frame -> run Florence captioning -> save caption + caption embedding
    label_embeddings = create_label_embeddings(flo_model, processor, csv)
    targets = list(label_embeddings.keys())
    images = get_images(video_path, timestamps)
    
    # MORE_DETAILED
    prompt = "<MORE_DETAILED_CAPTION>"
    timestamp_dict = get_timestamp_captions(flo_model, processor, images, timestamps, prompt)
    caption_embeddings = create_caption_embeddings(flo_model, processor, timestamp_dict, images)
    
    # MORE_DETAILED
    prompt = "<DENSE_REGION_CAPTION>"
    timestamp_dict = get_timestamp_captions(flo_model, processor, images, timestamps, prompt)
    region_embeddings = {}
    for timestamp_ms in timestamps:
        try:
            image = images[timestamp_ms]
            rr = timestamp_dict[timestamp_ms]
            if isinstance(rr, dict) and ('<DENSE_REGION_CAPTION>' in rr or 'bboxes' in rr):
                if '<DENSE_REGION_CAPTION>' in rr:
                    rr_inner = rr['<DENSE_REGION_CAPTION>']
                else:
                    rr_inner = rr
                bboxes = rr_inner['bboxes']
                labels = rr_inner['labels']
                if len(bboxes) and len(labels):
                    region_embeddings[timestamp_ms] = extract_region_and_label_embeddings(flo_model, processor, image, bboxes, labels)
        except Exception as e:
            print(f"[WARN] failed to extract region embeddings for {timestamp_ms}: {e}")

    # 4) Similarity computations
    cross = True
    if cross:
        embedding_similarity_matrix = calculate_similarity(label_embeddings, 
                                                                    caption_embeddings, 
                                                                    cross_similarity, 
                                                                    category='embedding', 
                                                                    label_level='def_embedding',
                                                                    normalize=True
                                                                    )
        df_embedding = get_df(caption_embeddings, label_embeddings, targets, embedding_similarity_matrix)
        visualize_cross_similarity(df_embedding.astype(float), "out_frame/emb_similarity_heatmap.png", red_labels=frame_annotations)
    
    label_cross = True
    if label_cross:
        embedding_similarity_matrix = calculate_similarity(label_embeddings, 
                                                                    caption_embeddings, 
                                                                    cross_similarity, 
                                                                    category='embedding', 
                                                                    label_level='label_embedding',
                                                                    normalize=True
                                                                    )
        df_embedding = get_df(caption_embeddings, label_embeddings, targets, embedding_similarity_matrix)
        visualize_cross_similarity(df_embedding.astype(float), "out_frame/lab_similarity_heatmap.png", red_labels=frame_annotations)

    next_word = False
    if next_word:    
        text_similarity_matrix = calculate_similarity(label_embeddings, 
                                                    caption_embeddings, 
                                                    next_word_similarity, 
                                                    category='caption', 
                                                    label_level='definition'
                                                    )
        df_text = get_df(caption_embeddings, label_embeddings, targets, text_similarity_matrix)
        # Load the CSV
        visualize_cross_similarity(df_text.astype(float), "out_frame/text_similarity_heatmap.png", red_labels=frame_annotations)
    
    region = False
    if region:
        similarity_matrix = {}
        for timestamp_ms in timestamps:
            similarities_labels = {}
            for label, embeddings in label_embeddings.items():
                label_element = embeddings['def_embedding']
                
                similarities = {}
                for bb_label in region_embeddings[timestamp_ms].keys():
                    embedding_dict = region_embeddings[timestamp_ms][bb_label]
                    bb_embedding = embedding_dict['label_embedding']
                    emb_similarity = cross_similarity(label_element, bb_embedding.squeeze(dim=0))
                    similarities[bb_label] = emb_similarity
                
                similarities_labels[label] = similarities
            
            similarity_matrix[timestamp_ms] = similarities_labels
    
    joint = True
    if joint:
        embedding_similarity_matrix = calculate_similarity(label_embeddings, 
                                                                    caption_embeddings, 
                                                                    cross_similarity, 
                                                                    category='joint_embedding', 
                                                                    label_level='def_embedding',
                                                                    normalize=True
                                                                    )
        df_embedding = get_df(caption_embeddings, label_embeddings, targets, embedding_similarity_matrix)
        visualize_cross_similarity(df_embedding.astype(float), "out_frame/joint_similarity_heatmap.png", red_labels=frame_annotations)

    for frame, image in images.items():
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(f'out_frame/{frame}.png')