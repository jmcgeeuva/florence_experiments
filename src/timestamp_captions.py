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
from .helper import get_frame_at_timestamp, run_task

def get_timestamp_captions(model, processor, video_path, timestamps, task_prompt):
    timestamp_dict = {}
    for timestamp_ms in timestamps:
        try:
            image = get_frame_at_timestamp(video_path, timestamp_ms)
            result = run_task(model, processor, task_prompt, image)  # Assume run_example accepts image input
            timestamp_dict[timestamp_ms] = result[task_prompt]
        except Exception as e:
            print(f"Skipping {video_file} at {timestamp_ms}ms due to error: {e}")
    
    return timestamp_dict

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

    timestamp_dict = get_timestamp_captions(model, processor, args.video, args.nargs, args.prompt)
    print_timestamp_captions(args.video, args.nargs, timestamp_dict)

