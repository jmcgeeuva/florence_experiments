from transformers import AutoProcessor, AutoModelForCausalLM
print(AutoModelForCausalLM.__module__)
from PIL import Image
import requests
import torch

import matplotlib.pyplot as plt  
import matplotlib.patches as patches  

from PIL import Image, ImageDraw, ImageFont 
import random
import numpy as np
import copy
import os
import csv


def run_example(model, processor, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    #print(f"inputs ---> {inputs}")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    #print(f"generated_ids ---> {generated_ids}")

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    #print(f"generated_text ---> {generated_text}")

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--label_path", default="/standard/spencerNSF/NeuralNetworksProjectVideos/Data Science Competition/Bounding_McGee/frames/book", help="The name of the user to greet.")
    parser.add_argument('--nargs', nargs='+', default=[3000, 3250, 3500, 4000])
    parser.add_argument('--outdir', default="detailed_captions/book")
    args = parser.parse_args()
    
    # Define model ID
    model_id = "microsoft/Florence-2-base-ft"

    # Load model and processor
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    book_dir = "/standard/spencerNSF/NeuralNetworksProjectVideos/Data Science Competition/Bounding_McGee/frames/book"
    output_dir = "detailed_captions/book"

    # def get_captions(book_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for folder_name in sorted(os.listdir(book_dir)):
        folder_path = os.path.join(book_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip files, only process folders

        output_csv_path = os.path.join(output_dir, f"{folder_name}.csv")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_num', 'caption'])

            print(f"Processing folder {folder_name}...")

            for image_name in sorted(os.listdir(folder_path)):
                if not image_name.endswith('.jpg'):
                    continue

                image_path = os.path.join(folder_path, image_name)

                try:
                    image = Image.open(image_path)

                    task_prompt = '<MORE_DETAILED_CAPTION>'
                    result = run_example(model, processor, task_prompt)

                    image_number = os.path.splitext(image_name)[0]

                    writer.writerow([image_number, result])

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    main()