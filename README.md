# florence_experiments


**Script**: create_embeddings.py
**Description**: Loops through the labels and their definitions to create pytorch embeddings for each label and its definition. Creates an embedding dictionary with labels as keys and values of label embeddings, definitions, and definition embeddings. This then is used to create a data file for each of the labels
**Options**:
* --csv: The csv label_definition file
* --output_dir: Where to save the pytorch files to
* --labels: The list of labels

**Script**: timestamp_captions.py
**Description**: This script is used to create a CSV named after the video that contains the name of the video with two columns image number and caption result. The first column has the video name and millisecond that the caption is taken from and the second contains the caption. 
**Options**:
* --video: path to the video to process
* --nargs: list of milliseconds in the video to process
* --prompt: The prompt to use including
    * <MORE_DETAILED_CAPTION>
    * <DETAILED_CAPTION>
    * <CAPTION>

**Script**: eaf_labels.py
**Description**: Creates a CSV file that contains the labels that occur at specific time frames in the video
**Options**:
* --video: path to the video to process
* --nargs: list of milliseconds in the video to process

**Script**: one_frame_labels.py
**Description**: This script creates a CSV with two columns: videoname with milliseconds and bounding boxes with labels. This is done by finding the MORE_DETAILED_CAPTION and then running caption_to_phrase_grounding on it
**Options**:
* --video: path to the video to process
* --nargs: list of milliseconds in the video to process

**Script**: computations.py
**Description**: Creates CSV files for the cross similarity and next word similarities of the caption data (pt files)
**Options**:
* --eaf: path to the eaf to pt directory
* --caption: path to the detailed caption embedding directory

**Script**: visualize.py
**Description**: Creates a visual of the attention scores from a cross_similarity_heatmap.csv (created by computations.py) file containing columns of timestamps of rows of each label and the score for that label
**Options**:
* --video: path to the video to process

**Script**: captions.py
**Description**: Create CSV files for each video in a label directory listing in detail the caption for each frame jpg in that directory
**Options**:
* --label_path: Path to directory of jpgs
* --nargs: List of timestamps
* --outdir: output directory to print to