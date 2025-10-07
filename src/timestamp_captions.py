import cv2

from transformers import AutoProcessor, AutoModelForCausalLM
import requests
import torch
from florence_pytorch.florence.modeling_florence2 import load

from PIL import Image, ImageDraw, ImageFont 
import random
import numpy as np
import copy
import os
import csv
import argparse
from src.helper import get_frame_at_timestamp, run_florence_task, get_flo_embeddings, get_joint_embedding

def get_images(video_path, timestamps):
    images = {}
    for timestamp_ms in timestamps:
        try:
            image = get_frame_at_timestamp(video_path, timestamp_ms)
            images[timestamp_ms] = image
        except Exception as e:
            print(f"Skipping {video_file} at {timestamp_ms}ms due to error: {e}")
    return images

def get_timestamp_captions(model, processor, images, timestamps, task_prompt):
    timestamp_dict = {}
    for timestamp_ms in timestamps:
        image = images[timestamp_ms]
        result = run_florence_task(model, processor, task_prompt, image)  # Assume run_example accepts image input
        timestamp_dict[timestamp_ms] = result[task_prompt]
    
    return timestamp_dict

def create_caption_embeddings(flo_model, processor, timestamp_dict, images=None):
    caption_dict = {}
    for timestamp, caption in timestamp_dict.items():
        caption_embedding = get_flo_embeddings(flo_model, processor, caption)
        if images is not None:
            joint_embedding = get_joint_embedding(flo_model, processor, caption, images[timestamp])
        else:
            joint_embedding = None
        
        caption_dict[timestamp] = {
            'caption': caption,
            'embedding': caption_embedding,
            'joint_embedding': joint_embedding
        }

    return caption_dict

def extract_region_and_label_embeddings(flo_model, processor, image: Image.Image, bboxes: list, labels: list):
    """
    For each bbox/label: crop the image region, run processor and attempt to extract embeddings
    Returns list of saved paths
    """
    region_embedding_dict = {}
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x0, y0, x1, y1 = bbox
        # ensure ints
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        cropped = image.crop((x0, y0, x1, y1)).convert("RGB")
        cropped_resized = cropped.resize((224, 224))

        # prepare processor inputs
        # NOTE: your embeddings.py used processor(images=..., return_tensors="pt") and then used 'input_ids' from it.
        img_tensor = processor(images=cropped_resized, return_tensors="pt")
        # If processor returned tokenized 'input_ids' (Florence sometimes tokenizes image tokens in multimodal mode)
        img_embedding = None
        if 'input_ids' in img_tensor:
            # move to device for model
            device = flo_model.device if hasattr(flo_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            img_ids = img_tensor['input_ids'].to(device)
            with torch.no_grad():
                try:
                    img_embedding = flo_model.get_input_embeddings()(img_ids)
                except Exception as e:
                    print(f"[WARN] could not get input embeddings for image tokens: {e}")
                    img_embedding = None
        else:
            # If there are pixel_values, try a model forward to attempt to get representations
            if 'pixel_values' in img_tensor:
                device = flo_model.device if hasattr(flo_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pv = img_tensor['pixel_values'].to(device)
                with torch.no_grad():
                    # NOTE: depending on Florence version, you might be able to call flo_model.get_image_features or pass pixel_values to the model
                    # We'll attempt several common options. If none work, img_embedding will be left None.
                    try:
                        # try image encoder method
                        img_feats = None
                        if hasattr(flo_model, 'get_image_features'):
                            img_feats = flo_model.get_image_features(pixel_values=pv)
                        else:
                            # attempt a forward pass to get hidden states
                            out = flo_model(pixel_values=pv, output_hidden_states=True)
                            if hasattr(out, 'last_hidden_state'):
                                img_feats = out.last_hidden_state.mean(dim=1)
                            elif 'hidden_states' in out:
                                img_feats = out.hidden_states[-1].mean(dim=1)
                        if img_feats is not None:
                            img_embedding = img_feats.cpu()
                    except Exception as e:
                        print(f"[WARN] image forward/featurization attempt failed: {e}")
                        img_embedding = None

        # embeddings.py
        label_tokens = processor.tokenizer(text=label, return_tensors="pt")
        device = flo_model.device if hasattr(flo_model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        label_input_ids = label_tokens['input_ids'].to(device)
        with torch.no_grad():
            try:
                label_embedding = flo_model.get_input_embeddings()(label_input_ids).cpu()
            except Exception as e:
                print(f"[WARN] could not get label embedding via get_input_embeddings(): {e}")
                label_embedding = None

        region_embedding_dict[f'{label.replace(' ', '_')}_{i}'] = {
            'bbox_index': i,
            'label': label,
            'cropped_image_embedding': img_embedding.detach().cpu() if isinstance(img_embedding, torch.Tensor) else img_embedding,
            'label_embedding': label_embedding.detach().cpu() if isinstance(label_embedding, torch.Tensor) else label_embedding
        }

    return region_embedding_dict

def print_timestamp_captions(video_path, timestamps, timestamp_dict):
    output_csv = video_path.split('/')[-1].replace('.mp4', 'timestamp.csv')
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_number", "caption_result"])
        for timestamp_ms in timestamps:
            result = timestamp_dict[timestamp_ms]
            image_number = f"{video_path.split('/')[-1]}_{timestamp_ms}ms"
            writer.writerow([image_number, result])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video", default="/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Videos_314 Hours/110.006.2018_ELA2_Year2_20180205.mp4", help="The name of the user to greet.")
    parser.add_argument('--nargs', nargs='+', default=[3000, 3250, 3500, 4000])
    parser.add_argument('--prompt', default="<MORE_DETAILED_CAPTION>")
    args = parser.parse_args()
    
    model_id = "BASE_FT" #"microsoft/Florence-2-base-ft"
    model, processor = load(model_id)

    images = get_images(args.video, args.nargs)
    timestamp_dict = get_timestamp_captions(model, processor, images, args.nargs, args.prompt)
    print_timestamp_captions(args.video, args.nargs, timestamp_dict)

