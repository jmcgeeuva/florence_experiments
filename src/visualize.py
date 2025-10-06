import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def visualize_cross_similarity(df, outfile, red_labels=None):
    # Convert columns to integers (time in ms)
    df.columns = df.columns.astype(int)

    # Create the heatmap
    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Similarity Score'})

    plt.xlabel("Time of Caption (ms)")
    plt.ylabel("Label (Definition)")
    plt.title("Cross-Similarities Between Detailed Caption and Definitions of Labels in Frame Heatmap")
    plt.tight_layout()

    # Customize y-axis tick label colors
    # if red_labels is not None:
    #     for tick_label in ax.get_yticklabels():
    #         label_text = tick_label.get_text()
    #         if label_text in red_labels:
    #             tick_label.set_color("red")
    #         else:
    #             tick_label.set_color("black")
    red_cells = []
    for frame, labels in red_labels.items():
        for label in labels:
            red_cells.append((frame, label))

    for i, row_label in enumerate(df.index):
        for j, col_label in enumerate(df.columns):
            val = df.iloc[i, j]
            color = "red" if red_cells and (col_label, row_label) in red_cells else "black"
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                    ha='center', va='center', color=color)

    plt.savefig(outfile, dpi=300)
    print('Done creating cross_similarity_heatmap')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video", default="./results/computations/cross_similarity_heatmap.csv", help="The name of the user to greet.")
    parser.add_argument("--out", default="cross_similarity_heatmap.png", help="The name of the user to greet.")    
    args = parser.parse_args()
    
    # Load the CSV
    df = pd.read_csv(args.video, index_col=0)
    visualize_cross_similarity(df, args.out)