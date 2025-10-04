from src.similarities import cross_similarity
from src.create_embeddings import create_embeddings
from src.visualize import visualize_cross_similarity
from src.computations import get_captions, next_word_similarity, calculate_similarity, create_table
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
    caption_file_path = "./embeddings/detailed_caption_embeddings/*.pt"
    caption_embeddings = get_captions(caption_file_path, args.targets)
    
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
    df_embedding.columns = df_embedding.columns.astype(float)
    df_text.columns = df_text.columns.astype(float)

    visualize_cross_similarity(df_embedding, "emb_similarity_heatmap.png")
    visualize_cross_similarity(df_text, "text_similarity_heatmap.png")