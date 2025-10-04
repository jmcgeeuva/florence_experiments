from florence_pytorch.florence.configuration_florence2 import *
from florence_pytorch.florence.florence_attn import *
import florence_pytorch.florence.modeling_florence2 as flor2
from florence_pytorch.florence.modeling_florence2 import load
from florence_pytorch.florence.processor import *
import csv
import cv2
from PIL import Image, ImageDraw, ImageFont 
# from florence_pytorch.florence.utils import run_example
import argparse
from src.helper import get_frame_at_timestamp, run_task

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video", default="/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Videos_314 Hours/110.006.2018_ELA2_Year2_20180205.mp4", help="The name of the user to greet.")
    parser.add_argument('--nargs', nargs='+', default=[3000, 3250, 3500, 4000])
    # parser.add_argument('--prompt', default="<MORE_DETAILED_CAPTION>")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model_id = "BASE_FT" #"microsoft/Florence-2-base-ft"
    model, processor = load(model_id, device)

    video_file = args.video
    output_csv = video_file.split('/')[-1].replace('.mp4', 'one_frame.csv')
    timestamps = args.nargs

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_number", "caption_result"])

        video_path = video_file
        for timestamp_ms in timestamps:
            try:
                image = get_frame_at_timestamp(video_path, timestamp_ms)
                task_prompt = "<MORE_DETAILED_CAPTION>"
                result = run_task(model, processor, task_prompt, image)[task_prompt]
                print("result phrase:", result)  
                task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
                result_2 = result = run_task(model, processor, task_prompt, image, text_input=result)
                image_number = f"{video_path.split('/')[-1]}_{timestamp_ms}ms"
                writer.writerow([image_number, result_2])
            except Exception as e:
                print(f"Skipping {video_file} at {timestamp_ms}ms due to error: {e}")


    # sentence = "A book is used or held by teacher or student"

    # tokens = processor.tokenizer(sentence)
    # t = torch.tensor(tokens['input_ids']).to(device=model.device)
    # embedding = model.get_input_embeddings()(t)
    # print(f'The tokens are as follows: {embedding} of size {embedding.shape} this means there are {embedding.shape[0]-2} words with 2 tokens for start and end and an embedding size of {embedding.shape[1]} from florence')
    return 0

if __name__ == '__main__':
    main()