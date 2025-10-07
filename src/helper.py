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
from florence_pytorch.florence.modeling_florence2 import shift_tokens_right
import os


colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
def draw_polygons(image, prediction, fill_mask=False):  
    """  
    Draws segmentation masks with polygons on an image.  
  
    Parameters:  
    - image_path: Path to the image file.  
    - prediction: Dictionary containing 'polygons' and 'labels' keys.  
                  'polygons' is a list of lists, each containing vertices of a polygon.  
                  'labels' is a list of labels corresponding to each polygon.  
    - fill_mask: Boolean indicating whether to fill the polygons with color.  
    """  
    # Load the image  
   
    draw = ImageDraw.Draw(image)  
      
   
    # Set up scale factor if needed (use 1 if not scaling)  
    scale = 1  
      
    # Iterate over polygons and labels  
    for polygons, label in zip(prediction['polygons'], prediction['labels']):  
        color = random.choice(colormap)  
        fill_color = random.choice(colormap) if fill_mask else None  
          
        for _polygon in polygons:  
            _polygon = np.array(_polygon).reshape(-1, 2)  
            if len(_polygon) < 3:  
                print('Invalid polygon:', _polygon)  
                continue  
              
            _polygon = (_polygon * scale).reshape(-1).tolist()  
              
            # Draw the polygon  
            if fill_mask:  
                draw.polygon(_polygon, outline=color, fill=fill_color)  
            else:  
                draw.polygon(_polygon, outline=color)  
              
            # Draw the label text  
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  
  
    # Save or display the image  
    #image.show()  # Display the image  
    image.save("image4.png")

def draw_polygons_on_image(image: Image.Image, prediction: dict, out_path: str, fill_mask=False):

    img = image.copy()
    draw = ImageDraw.Draw(img)
    if 'bboxes' in prediction and 'labels' in prediction:
        for bbox, label in zip(prediction['bboxes'], prediction['labels']):
            x0, y0, x1, y1 = map(int, bbox)
            color = random.choice(COLORMAP)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x0 + 4, y0 + 4), label, fill=color)
    elif 'polygons' in prediction and 'labels' in prediction:
        for polygons, label in zip(prediction['polygons'], prediction['labels']):
            color = random.choice(COLORMAP)
            for _polygon in polygons:
                _polygon = np.array(_polygon).reshape(-1, 2).tolist()
                draw.polygon(sum([_polygon], []), outline=color)
            if len(polygons) and len(polygons[0]):
                px, py = polygons[0][0][0], polygons[0][0][1]
                draw.text((px + 4, py + 4), label, fill=color)
    img.save(out_path)
    print(f"Saved annotated image {out_path}")

def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.savefig("image2.png")

def get_frame_at_timestamp(video_path, timestamp_ms):
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
    return Image.fromarray(frame_rgb)

def get_flo_embeddings(flo_model, processor, element):
    tokens = processor.tokenizer(element)
    tensor = torch.tensor(tokens['input_ids']).to(device=flo_model.device)
    embedding = flo_model.get_input_embeddings()(tensor)
    return embedding

def get_joint_embedding(flo_model, processor, caption_text, image):
    # Try to get a joint image-text embedding if model supports it
    joint_embedding = None
    # try:
    inputs = processor(text=caption_text, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(flo_model.device) for k,v in inputs.items()}
    with torch.no_grad():
        decoder_input_ids = shift_tokens_right(inputs['input_ids'].to(dtype=int), flo_model.config.pad_token_id, 0)
        out = flo_model(input_ids=inputs['input_ids'].to(dtype=int), pixel_values=inputs['pixel_values'], output_hidden_states=True, decoder_input_ids=decoder_input_ids.to(device=flo_model.device, dtype=int))
        import pdb; pdb.set_trace()
        # attempt to build joint by averaging last hidden states if present
        joint_embedding = out.encoder_last_hidden_state.squeeze(dim=0) #mean(dim=1).cpu()
    # except Exception as e:
    #     print(f"[WARN] joint image-text embedding attempt failed: {e}")
    #     joint_embedding = None
    return joint_embedding


def run_florence_task(model, processor, task_prompt: str, image: Image.Image, text_input: str = None):
    """
    Run Florence multimodal generation.
    Returns parsed post-processed result (dictionary keyed by task_prompt)
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )


    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer