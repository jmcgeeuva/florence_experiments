import os
from PIL import Image
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import random

class MultiTHUMOSDataset(Dataset):
    def __init__(self, dataset_path, annotations, multi=True, split='val', train=False, label_csv='labels.csv', fps=30, transform=None, size=768, stunt=1):
        """
        Args:
            dataset_path: root directory containing videos and annotations
            split: 'val' or 'test'
            label_csv: path to label CSV listing all action classes
            fps: assumed frames per second (to map timestamps to frame indices)
            transform: optional torchvision transforms for image preprocessing
        """
        self.dataset_path = dataset_path
        self.train = train
        self.multi_annotations_dir = annotations
        self.annotations_dir = os.path.join(dataset_path, f'annotations_{split}')
        self.videos_dir = os.path.join(dataset_path, 'videos')
        self.fps = fps
        self.transform = transform
        self.size=size

        # Read label names
        with open('./multilabels.csv', 'r') as f:
            reader = csv.reader(f)
            self.labels = [row[0] for row in reader]
        self.num_labels = len(self.labels)

        thumos_test = {
            'test': [], 'val': []
        }
        with open('./labels.csv', 'r') as l:
            reader = csv.reader(l)
            for split in ['val', 'test']:
                for row in reader:
                    ann_path = os.path.join(self.annotations_dir, f"{row[0]}_{split}.txt")
                    with open(ann_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 3:
                                continue
                            video_name, start_time, end_time = parts
                            thumos_test[split].append(video_name)

        if self.train:
            self.split = 'test'
        else:
            self.split = 'val'

        # Build annotation index: {video: [(start_frame, end_frame, class_idx), ...]}
        self.video_annotations = {}
        for label_idx, label in enumerate(self.labels):
            ann_path = os.path.join(self.multi_annotations_dir, f"{label}.txt")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                    video_name, start_time, end_time = parts
                    if (self.train and 'test' not in video_name) or (not self.train and 'validation' not in video_name):
                        continue
                    #video_path = os.path.join(self.videos_dir, self.split, f"{video_name}.mp4")
                    #cap = cv2.VideoCapture(video_path)
                    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    #cap.release()
                    #if float(end_time) > total_frames:
                    #    import pdb; pdb.set_trace()
                    start_frame = float(start_time)
                    end_frame = float(end_time)
                    self.video_annotations.setdefault(video_name, []).append(
                        (start_frame, end_frame, label_idx)
                    )

        # Build a frame-level index for __getitem__
        self.frame_index = []  # list of (video_name, frame_idx)
        for video_name, annos in self.video_annotations.items():
            #start_list = sorted([start for start, _, _ in annos])
            oend_list = sorted([end for _, end, _ in annos])
            video_path = os.path.join(self.videos_dir, self.split, f"{video_name}.mp4")
            if not os.path.exists(video_path):
                continue
            frames = set()
            indices = list()
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for start, end, idx in annos:
                start = int(start * fps)
                end = int(end * fps)
                if start > total_frames:
                    continue
                if end > total_frames:
                    end = total_frames
                # steps of fps in between frames in order to not duplicate
                frame_range = list(range(start, end, int(fps)))
                frames = frames | set(frame_range)
            self.frame_index.extend([(video_name, frame) for frame in sorted(list(frames))])
            #self.frame_index = sorted(list(self.frame_index))
            #for frame_idx in range(start_list[0], end_list[-1]):
            #    self.frame_index.append((video_name, frame_idx))
        if stunt != 1:
            indices = range(len(self.frame_index))
            sampled_indexes = list(random.sample(indices, int(len(indices)*stunt)))
            self.frame_index = [info for i, info in enumerate(self.frame_index) if i in sampled_indexes]
        self.test = {}

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, idx, debug=False):
        video_name, frame_idx = self.frame_index[idx]
        if self.train:
            video_path = os.path.join(self.videos_dir, 'test', f"{video_name}.mp4")
        else:
            video_path = os.path.join(self.videos_dir, 'val', f"{video_name}.mp4")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            if debug:
                import pdb; pdb.set_trace()
            raise ValueError(f"Could not read frame {idx} {frame_idx} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = Image.fromarray(frame)
            frame = img.resize((self.size, self.size), Image.BILINEAR)
            frame = self.transform(frame)
        else:
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Create multi-hot label vector
        label_vec = np.zeros(self.num_labels, dtype=np.float32)
        for start, end, cls_idx in self.video_annotations.get(video_name, []):
            if start <= frame_idx <= end:
                label_vec[cls_idx] = 1.0
        label_vec = torch.from_numpy(label_vec)

        return frame, frame_idx, label_vec


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", default="/mnt/data_sdb/mmaction2/data/thumos14/", help="The name of the user to greet.")
    parser.add_argument("--annotations", default="/mnt/data_sdb/multithumos/annotations/")
    parser.add_argument("--split", default="val", help="The name of the user to greet.")
    parser.add_argument("--labels", default="multilabels.csv", help="The name of the user to greet.")
    parser.add_argument('--nargs', nargs='+', default=[3000, 3250, 3500, 4000])
    # parser.add_argument('--prompt', default="<MORE_DETAILED_CAPTION>")
    args = parser.parse_args()

    dataset_train = MultiTHUMOSDataset(
        dataset_path=args.dataset,
        annotations=args.annotations,
        train=True,
        label_csv=args.labels
    )
    dataset_val =  MultiTHUMOSDataset(dataset_path=args.dataset, annotations=args.annotations, train=False, label_csv=args.labels)

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=32)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=32)

    #dataset_train.__getitem__(923233, True)

    label_count = [0 for _ in range(len(dataset_train.labels))]
    for (images, timestamp, labels) in tqdm(train_loader, total=len(train_loader)):
        for label_list in labels:
            indices = [i for i, x in enumerate(label_list) if x == 1]
            for ind in indices:
                label_count[ind] += 1

    for i, label in enumerate(dataset_val.labels):
        print(f'Train label {label}: {label_count[i]}')

    label_count = [0 for _ in range(len(dataset_train.labels))]
    for (images, timestamp, labels) in tqdm(val_loader, total=len(val_loader)):
        for label_list in labels:
            indices = [i for i, x in enumerate(label_list) if x == 1]
            for ind in indices:
                label_count[ind] += 1

    for i, label in enumerate(dataset_val.labels):
        print(f'Test label {label}: {label_count[i]}')
