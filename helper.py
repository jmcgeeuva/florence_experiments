from florence_pytorch.florence.configuration_florence2 import *
from florence_pytorch.florence.florence_attn import *
import florence_pytorch.florence.modeling_florence2 as flor2
from florence_pytorch.florence.modeling_florence2 import load
from florence_pytorch.florence.processor import *
import csv
import cv2
from PIL import Image, ImageDraw, ImageFont 
from florence_pytorch.florence.utils import run_example
import argparse

import os

def get_frame_at_timestamp(video_path, timestamp_ms):
    print("Getting frame")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not retrieve frame at {timestamp_ms}ms from {video_path}")

    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("got frame")
    return Image.fromarray(frame_rgb)

def run_task(model, processor, task_prompt, image, text_input=None):
    print("running example")
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    print("getting inputs")
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

    print("got inputs")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    #print(f"generated_ids ---> {generated_ids}")
    print("generated ids")
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    #print(f"generated_text ---> {generated_text}")
    print("generated text")
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    print("ran example")

    return parsed_answer