import torch.utils.data as data
import glob
import os.path as osp
from src.timestamp_captions import get_timestamp_captions, create_caption_embeddings, get_images, extract_region_and_label_embeddings
from src.eaf_labels import get_eaf_labels
import pandas as pd
import torch
import random
import cv2
from PIL import Image, ImageDraw, ImageFont 
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from speach import elan
from datetime import datetime
import itertools
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def get_actions_every_30s(eaf_path, step_ms=30000):
    """
    Reads an ELAN .eaf file and returns a list of annotations
    active at every 30-second interval (or any given step in ms).
    """
    eaf = elan.read_eaf(eaf_path)
    actions_at_steps = []

    # Find the total duration based on all annotations
    all_ends = [ann.to_ts.ts for tier in eaf for ann in tier]
    if all_ends:
        max_ts = max(all_ends)
    else:
        raise ValueError(f'[WARNING] {eaf_path} is empty!')
    max_start = datetime.strptime(max_ts, "%H:%M:%S.%f")
    milliseconds_max = (max_start.hour * 3600 + max_start.minute * 60 + max_start.second) * 1000 + int(max_start.microsecond / 1000)

    # Loop over every 30-second step
    for t in range(0, int(milliseconds_max) + step_ms, step_ms):
        current_actions = []
        for tier in eaf:
            for ann in tier:
                # Check if this annotation is active at timestamp t
                ann_start = ann.from_ts.ts
                ann_end = ann.to_ts.ts
                ann_end = datetime.strptime(ann_end, "%H:%M:%S.%f")
                ann_start = datetime.strptime(ann_start, "%H:%M:%S.%f")
                milliseconds_start = (ann_start.hour * 3600 + ann_start.minute * 60 + ann_start.second) * 1000 + int(ann_start.microsecond / 1000)
                milliseconds_end = (ann_end.hour * 3600 + ann_end.minute * 60 + ann_end.second) * 1000 + int(ann_end.microsecond / 1000)
                if milliseconds_start <= t <= milliseconds_end:
                    current_actions.append({
                        "tier": tier.ID,
                        "start": ann.from_ts.ts,
                        "end": ann.to_ts.ts,
                        "text": ann.text
                    })

        actions = [t['tier'] for i, t in enumerate(current_actions) if (i != 0 and t['tier'] not in ['Student Location', 'Representing Content', 'Teacher Location', 'Camera Actions'] and t['text'] != 'moving')]
        actions_at_steps.append({
            "timestamp_ms": t,
            "actions": actions if actions != [] else ["No Action"]
        })

    return actions_at_steps

# Preprocess the description list and file dictionary to preload all the data locations and timestamps into a dictionary
# This way we will know exactly how many examples with have and will be able to load more data systematically
# FIXME move preprocessing to outside the class and then have this as input into the class (do the train/test split outside too)
def load_file_dict(video_path, annot_path):
    vid_glob = glob.glob(osp.join(video_path, "*.mp4"))
    eaf_glob = glob.glob(osp.join(annot_path, "*.eaf"))
    eafs = [eaf.split("/")[-1] for eaf in eaf_glob]
    vids = [vid.split("/")[-1] for vid in vid_glob]
    descriptors = set([eaf.replace('.eaf', '') for eaf in eafs]) & set([eaf.replace('.mp4', '') for eaf in vids])
    file_dict = OrderedDict({desc: {ext: osp.join(path, desc+'.'+ext) for path, ext in [(video_path, 'mp4'), (annot_path, 'eaf')]} for desc in descriptors})
    for desc in list(file_dict.keys()):
        vid = file_dict[desc]['mp4']
        annot = file_dict[desc]['eaf']
        try:
            frame_annotations = get_actions_every_30s(annot)
            file_dict[desc]['timestamps'] = [action['timestamp_ms'] for action in frame_annotations]
            file_dict[desc]['actions'] = [action['actions'] for action in frame_annotations]
        except:
            print(f'[WARNING] Removing {desc} because the eaf is empty')
            del file_dict[desc]
            continue
    desc_list = list(file_dict.keys())
    return file_dict

class EducationDataset(data.Dataset):
    def __init__(self, video_path, annot_path, file_dict, transform=None, train=True, seed=42, train_ratio=0.8):
        self.video_path = video_path
        self.annot_path = annot_path
        self.fps = 30
        self.time_frame = 900 # every 30 minutes
        self.transform = transform
        self.train = train
        self.file_dict = file_dict


        df = pd.read_csv('./csv/label_definitions.csv', header=None, names=['label', 'definition'], engine='python')
        self.gt_labels = [(row['label'], row['definition']) for index, row in df.iterrows() if index!=0]
        
        # deterministic split
        desc_list = list(self.file_dict.keys())
        np.random.seed(seed)
        indices = np.arange(len(desc_list))
        np.random.shuffle(indices)
        split_idx = int(train_ratio * len(indices))

        if train:
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]
        self.desc_list = [desc_list[ind] for ind in indices]
        cumsum = 0
        tmp = {}
        for desc in self.desc_list:
            length = len(self.file_dict[desc]['actions'])
            tmp[desc] = self.file_dict[desc]
            tmp[desc]['len'] = length
            tmp[desc]['cumsum'] = cumsum
            cumsum += length
        self.file_dict = tmp

    def generate_ground_truth(self, labels, gt_labels):
        all_labels = [label.lower() for label, _ in gt_labels]
        label_tensor = []
        corrections = {
            'Carpet or floor-sitting': 'Sitting on carpet or floor',
            'Standing (t)': "Standing",
            'Desks-sitting': "Sitting at desks",
            'Whole class activity': "Whole-class activity",
            'Instructional tool-using or holding': "Using or holding instructional tool",
            'Raising hand': "Student raising hand",
            'Multiple without ss interaction': "Multiple students without student interaction",
            'Group tables-sitting': "Sitting at group tables",
            'Worksheet-using or holding': "Using or holding worksheet",
            'Book-using or holding': "Using or holding book",
            'Multiple with ss interaction': "Multiple students with student interaction",
            "Notebook-using or holding": "Using or holding notebook"
        }
        
        new_labels = []
        tensor = torch.zeros(len(all_labels))
        for i, label in enumerate(labels):
            for key, value in corrections.items():
                if key.lower() == label.lower():
                    label = value
                    break
            if label.lower() not in all_labels:
                print(f'[WARNING] "{label}" are not in the list of all labels')
            else:
                new_labels.append(label)

            idx = all_labels.index(label.lower())
            tensor[idx] = 1

        return tensor
    
    def get_images(self, video_path, timestamps):
        images = []
        for timestamp_ms in timestamps:
            try:
                image = self.get_frame_at_timestamp(video_path, timestamp_ms//(1000))
                images.append(image)
            except Exception as e:
                raise ValueError(f"Skipping {video_path} at {timestamp_ms}ms due to error: {e}")
        return images

    def get_frame_at_timestamp(self, video_path, timestamp_ms):
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
    



    def get_file_dict(self):
        return self.file_dict

    def __len__(self):
        return sum([value['len'] for key, value in self.file_dict.items()])

    def __getitem__(self, index, choice=None):
        # get closest number to index that is still lower
        cumsum = [value['cumsum'] for key, value in self.file_dict.items()]
        index_val = max([n for n in cumsum if n <= index])
        # Get index of that number and then find the descriptor
        dict_index = cumsum.index(index_val)
        desc = list(self.file_dict.keys())[dict_index]

        # Retrieve file locations and timestamp/action pair
        length = self.file_dict[desc]['len']
        cumsum = self.file_dict[desc]['cumsum']
        annot_file = self.file_dict[desc]['eaf']
        vid_file   = self.file_dict[desc]['mp4']
        timestamp  = self.file_dict[desc]['timestamps'][cumsum - index_val]
        actions    = self.file_dict[desc]['actions'][cumsum - index_val]

        # frame_annotations = get_eaf_labels(osp.join(self.annot_path, annot_file), None)
        # get every 1800 (30 seconds) seconds get a frame 
        # print(choice)
        try:
            images = self.get_images(osp.join(self.video_path, vid_file), [timestamp])
        except:
            print(f'ERROR: {index} {choice}')
            raise
        labels = [label.strip() for label in actions]
        
        # Create ground truth for each frame
        label_tensor = self.generate_ground_truth(labels, self.gt_labels)

        if self.transform:
            ret_img_group = [img.resize((224, 224), Image.BILINEAR) for img in images]
            images = self.transform(ret_img_group[0])

        
        return images, timestamp, label_tensor

if __name__ == "__main__":
    vid_dir = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Videos_314 Hours/"
    annot_dir = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Video Annotations_314 Hours/"
    batch_size = 32
    workers = 3

    file_dict = load_file_dict(vid_dir, annot_dir)
    edu = EducationDataset(vid_dir, annot_dir, file_dict, transform=transforms.ToTensor())
    edu_test = EducationDataset(vid_dir, annot_dir, file_dict, transform=transforms.ToTensor(), train=False)

    print(f'There are {len(list(edu.get_file_dict().keys()))} video-eaf pairs')

    image, timestamp, labels = edu[0]

    def collate_fn(batch):
        image, timestamp, label = zip(*batch)
        # Check the labels for bb
        image = torch.stack(image) 
        timestamps = torch.tensor(timestamp)
        labels = pad_sequence(label, batch_first=True, padding_value=-1)
        return image, timestamp, labels

    train_loader = DataLoader( edu, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False, drop_last=True,  collate_fn=collate_fn)
    test_loader = DataLoader( edu_test, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=False, drop_last=True,  collate_fn=collate_fn)

    print(f'There are {len(train_loader)*batch_size} samples')
    print(f'There are {len(test_loader)*batch_size} samples')

    label_count = [0 for _ in range(27)]
    for (images, timestamp, labels) in tqdm(train_loader, total=len(train_loader)):
        for label_list in labels: 
            indices = [i for i, x in enumerate(label_list) if x == 1]
            for ind in indices:
                label_count[ind] += 1

    for i, (label, definition) in enumerate(edu.gt_labels):
        print(f'Train label {label}: {label_count[i]}')

    label_count = [0 for _ in range(27)]
    for (images, timestamp, labels) in tqdm(test_loader, total=len(test_loader)):
        for label_list in labels: 
            indices = [i for i, x in enumerate(label_list) if x == 1]
            for ind in indices:
                label_count[ind] += 1

    for i, (label, definition) in enumerate(edu.gt_labels):
        print(f'Test label {label}: {label_count[i]}')