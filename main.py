from src.similarities import cross_similarity
from src.create_embeddings import create_embeddings
from src.visualize import visualize_cross_similarity
from src.computations import get_captions, next_word_similarity, calculate_similarity, create_table
from src.timestamp_captions import get_timestamp_captions, create_caption_embeddings
import florence_pytorch.florence.modeling_florence2 as flor2
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--csv", default='./csv/label_definitions.csv', help="The name of the user to greet.")
    parser.add_argument("--output_dir", default='./tmp/', help="The name of the user to greet.")
    parser.add_argument('--targets', nargs='+', default=['Individual activity', 'Student writing', 'Individual technology', 'Sitting at desks', 'Student(s) standing or walking', 'Small group activity', 'Student raising hand'])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    
    flo_model, processor = flor2.load("BASE_FT", device)

    label_embeddings = create_embeddings(flo_model, processor, args.csv)
    
    
    # FIXME Dictionary of frames with text caption and embedding
    task_prompt = "<MORE_DETAILED_CAPTION>"
    video_path = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Videos_314 Hours/110.006.2018_ELA2_Year2_20180205.mp4"
    timestamps = [3000] #, 3250, 4000]
    timestamp_dict = get_timestamp_captions(flo_model, processor, video_path, timestamps, task_prompt)
    caption_embeddings = create_caption_embeddings(flo_model, processor, timestamp_dict)
    
    embedding_similarity_matrix = calculate_similarity(label_embeddings, 
                                                                 caption_embeddings, 
                                                                 cross_similarity, 
                                                                 category='embedding', 
                                                                 label_level='def_embedding'
                                                                )
    text_similarity_matrix = calculate_similarity(label_embeddings, 
                                                  caption_embeddings, 
                                                  next_word_similarity, 
                                                  category='caption', 
                                                  label_level='definition'
                                                 )
    
    df_embedding, df_text, axis_labels = create_table(caption_embeddings, 
                                                      args.targets, 
                                                      label_embeddings, 
                                                      embedding_similarity_matrix, 
                                                      text_similarity_matrix
                                                     )
    
    
    # Load the CSV
    visualize_cross_similarity(df_embedding.astype(float), "emb_similarity_heatmap.png")
    visualize_cross_similarity(df_text.astype(float), "text_similarity_heatmap.png")