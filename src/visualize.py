import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def visualize_cross_similarity(df, outfile):
    # Create the heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Similarity Score'})

    plt.xlabel("Time of Caption (ms)")
    plt.ylabel("Label (Definition)")
    plt.title("Cross-Similarities Between Detailed Caption and Definitions of Labels in Frame Heatmap")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)

    print('Done creating cross_similarity_heatmap')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video", default="./results/csv/cross_similarity_heatmap.csv", help="The name of the user to greet.")
    parser.add_argument("--out", default="cross_similarity_heatmap.png", help="The name of the user to greet.")    
    args = parser.parse_args()
    
    # Load the CSV
    df = pd.read_csv(args.video, index_col=0)

    # Convert columns to integers (time in ms)
    df.columns = df.columns.astype(int)

    visualize_cross_similarity(args.video, args.out)