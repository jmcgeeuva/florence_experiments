from src.similarities import cross_similarity
from src.create_embeddings import create_embeddings as create_label_embeddings
from src.visualize import visualize_cross_similarity
from src.computations import get_captions, next_word_similarity, calculate_similarity, get_df
from src.timestamp_captions import get_timestamp_captions, create_caption_embeddings, get_images, extract_region_and_label_embeddings
from src.eaf_labels import get_eaf_labels
from datasets import EducationDataset, load_file_dict

import florence_pytorch.florence.modeling_florence2 as flor2
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from speach import elan
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from infonce import SupervisedInfoNCE
import torch.nn.functional as F
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def pad_with_preserved_tokens(seqs, start_token=0, end_token=2, pad_value=0):
    """
    seqs: list of 1D torch tensors (each includes start and end tokens)
    start_token: token to prepend back
    end_token: token to append back
    pad_value: value used for padding
    """
    # 1. Remove start and end tokens
    stripped = [s[1:-1] for s in seqs]

    # 2. Pad internal parts automatically
    padded_inner = pad_sequence(stripped, batch_first=True, padding_value=pad_value)
    B, L = padded_inner.shape

    # 3. Reattach start and end tokens
    start_col = torch.full((B, 1), start_token, dtype=torch.long, device=seqs[0].device)
    end_col = torch.full((B, 1), end_token, dtype=torch.long, device=seqs[0].device)

    padded = torch.cat([start_col, padded_inner, end_col], dim=1)
    return padded

def create_embeddings(flo_model, processor, gt_labels):
    embedding_data = []
    attention_mask = []
    # flo_model = flo_model.module
    for label, definition in gt_labels:
        # Process label
        tokens = processor.tokenizer(label)
        embedding_data.append(torch.tensor(tokens['input_ids'], device=flo_model.module.device))
        attention_mask.append(torch.tensor(tokens['attention_mask'], device=flo_model.module.device))

    input_ids = pad_with_preserved_tokens(embedding_data)
    inputs_embeds = torch.stack([flo_model.module.get_input_embeddings()(tensor) for tensor in input_ids])
    # 0 to ignore these masked tokens
    attention_mask = pad_with_preserved_tokens(attention_mask, pad_value=0)
    
    return input_ids, inputs_embeds, attention_mask
    
def create_img_and_label_embeddings(flo_model, processor, prompt, sample, gt_labels):
    prompts = [prompt for _ in sample]
    samples = [img for img in sample]
    
    # Create prompt/sample inputs
    # flo_model = flo_model.module
    inputs = processor(text=prompts, images=samples, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].to(device=flo_model.module.device)
    pixel_values = inputs['pixel_values'].to(device=flo_model.module.device)
    decoder_img_ids = flor2.shift_tokens_right(input_ids.to(dtype=int), flo_model.module.config.pad_token_id, 0).to(device=flo_model.module.device, dtype=int)
    img_out   = flo_model(input_ids = input_ids,
                        pixel_values = pixel_values,
                        decoder_input_ids = decoder_img_ids,
                        output_hidden_states = True)

    # Create all label embeddings in a list ordered by label name
    label_embeddings, label_embeddings_embeds, attention_mask = create_embeddings(flo_model, processor, gt_labels)
    
    decoder_label_ids = flor2.shift_tokens_right(label_embeddings.to(dtype=int), flo_model.module.config.pad_token_id, 0).to(device=flo_model.module.device, dtype=int)
    label_out = flo_model(inputs_embeds=label_embeddings_embeds, output_hidden_states=True, attention_mask=attention_mask, decoder_input_ids=decoder_label_ids)
    

    img_out_state = img_out.decoder_hidden_states[-1]
    lab_out_state = label_out.decoder_hidden_states[-1]
    img_out_mean = img_out_state.mean(dim=1)
    lab_out_mean = lab_out_state.mean(dim=1)
    return img_out_mean, lab_out_mean

def calculate_accuracy(matrix, labels):
    num = 0
    corr_k = 0
    labeled_ids = []
    correct_ids = []
    for mat, label_list in zip(matrix, labels):
        indices = [i for i, x in enumerate(label_list) if x == 1]
        values_k, indices_k = mat.topk(len(indices), dim=-1)
        for ind in indices:
            if ind in indices_k:
                corr_k += 1
            num += 1
        labeled_ids.append(indices_k)
        correct_ids.append(indices)
    return num, corr_k, labeled_ids, correct_ids

def plot_confusion_matrix(y_true, y_pred, classes, name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix

    new_pred = []
    new_true = []
    for i, (pred, gt) in enumerate(zip(y_pred, y_true)):
        # Find the predictions that were correct through the intersection
        intersection = set(y_pred) & set(y_true)
        if intersection:
            new_pred.extend(intersection)
            new_true.extend(intersection)
        # find the indices that are only in y_pred
        difference = set(y_pred) - set(y_true)
        if difference:
            new_pred.extend(difference)
            new_true.extend(set(y_true) - set(y_pred))

    y_pred = new_pred
    y_true = new_true
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    with open('confusion.txt', 'w') as f:
        for el in cm:
            for np_entry in el:
                f.write(f'{np_entry},')
            f.write('\n')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(f'{name}.png')
    plt.clf()

def validate(test_loader, flo_model, processor, prompt, gt_labels, loss_info, loss_ce, epoch, epochs):
    flo_model.eval()
    total_num = 0
    total_corr = 0
    labeled_ids = []
    correct_ids = []
    with torch.no_grad():
        for (images, timestamp, labels) in tqdm(test_loader, total=len(test_loader)):
            img_out_mean, lab_out_mean = create_img_and_label_embeddings(flo_model, processor, prompt, images, gt_labels)
            _, matrix = cross_similarity(img_out_mean, lab_out_mean)
            num, corr_k, labeled_ids, correct_ids = calculate_accuracy(matrix, labels)
            total_num += num
            total_corr += corr_k
                
            # Two lists of lists where each element of the list is a list of predicted labels and a list of correct labels
            labeled_ids.extend(labeled_ids)
            correct_ids.extend(correct_ids)

    plot_confusion_matrix(correct_ids, labeled_ids, np.array(gt_labels), 'test')
    
    print(f'Epoch: [{epoch+1}/{epochs}]: Top1: {(corr_k/num)*100}%')

def contrastive_info_nce(
    batch_embs: torch.Tensor,
    label_embs: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.07,
    normalize: bool = True,
    reduction: str = "mean",
    ignore_no_positive: bool = True) -> torch.Tensor:
    
    B, D = batch_embs.shape
    L = label_embs.shape[0]
    device = batch_embs.device

    if normalize:
        batch_embs = F.normalize(batch_embs, p=2, dim=1)
        label_embs = F.normalize(label_embs, p=2, dim=1)

    # logits: (B, L) similarities between each sample and each label
    logits = torch.matmul(batch_embs, label_embs.T) / float(temperature)

    # create boolean mask for positives
    pos_mask = targets.to(torch.bool)

    # check which rows have at least one positive
    has_pos = pos_mask.any(dim=1)  # (B,)
    if not torch.all(has_pos):
        if not ignore_no_positive:
            missing = (~has_pos).nonzero(as_tuple=False).squeeze(1).tolist()
            raise ValueError(f"Some samples have no positive labels (rows: {missing})")

    # compute logsumexp over all labels (denominator) -> shape (B,)
    logsumexp_all = torch.logsumexp(logits, dim=1)  # (B,)

    # mask out non-positive logits by setting them to -inf so they don't contribute to logsumexp
    neg_inf = -1e9  # using a large negative constant is numerically stable for logsumexp
    masked_logits_for_pos = logits.masked_fill(~pos_mask, neg_inf)  # (B, L)

    # For rows where all entries were masked (no positives), masked_logits_for_pos will be all -inf,
    # and logsumexp will produce -inf. We'll filter those out later if ignore_no_positive is True.
    logsumexp_pos = torch.logsumexp(masked_logits_for_pos, dim=1)  # (B,)

    per_sample_loss = -(logsumexp_pos - logsumexp_all)  # (B,)

    return per_sample_loss.mean()



def train(train_loader, test_loader, flo_model, processor, optimizer, prompt, gt_labels, loss_info, loss_ce):
    epochs = 50
    freq = 1
    for epoch in range(epochs):
        debug_count = 0
        for (images, timestamp, labels) in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            flo_model.train()

            img_out_mean, lab_out_mean = create_img_and_label_embeddings(flo_model, processor, prompt, images, gt_labels)
            _, matrix = cross_similarity(img_out_mean, lab_out_mean)
            labels = labels.to(device=matrix.device, dtype=torch.float)

            ce_loss = loss_ce(matrix, labels)
            info_loss = contrastive_info_nce(batch_embs=img_out_mean, label_embs=lab_out_mean, targets=labels, temperature=0.07)
            loss = .6*ce_loss + .4*info_loss
            loss.backward()
            optimizer.step()

            if debug_count % 1000 == 0:
                num, corr_k, _, _ = calculate_accuracy(matrix, labels)
                print(f'[{epoch+1}/{epochs}] Loss: {loss.item()}, Accuracy: {corr_k}/{num} = {(corr_k/num)*100}%')
                break
            debug_count += 1
        
        if epoch % freq == 0:
            validate(test_loader, flo_model, processor, prompt, gt_labels, loss_info, loss_ce, epoch, epochs)
    
    flo_model.eval()
    return flo_model

def get_actions_every_30s(eaf_path, step_ms=30000):
    """
    Reads an ELAN .eaf file and returns a list of annotations
    active at every 30-second interval (or any given step in ms).
    """
    eaf = elan.read_eaf(eaf_path)
    actions_at_steps = []

    # Find the total duration based on all annotations
    all_ends = [ann.to_ts.ts for tier in eaf for ann in tier]
    max_ts = max(all_ends) if all_ends else 0
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

        actions = [t['tier'] for i, t in enumerate(current_actions) if (i != 0 and t['tier'] not in ['Student Location', 'Representing Content'] and t['text'] != 'moving')]
        actions_at_steps.append({
            "timestamp_ms": t,
            "actions": actions if actions != [] else ["No Action"]
        })

    

    return actions_at_steps

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flo_model, processor = flor2.load("BASE_FT", device, lora=False)
    # flo_model = flo_model.to(device)

    flo_model = torch.nn.DataParallel(flo_model).cuda()
    
    # set up loss functions
    loss_ce = torch.nn.BCEWithLogitsLoss()
    loss_info = SupervisedInfoNCE(temperature=0.07)
    

    # video_path = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Videos_314 Hours/110.006.2018_ELA2_Year2_20180205.mp4"
    # annot_path = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Video Annotations_314 Hours/110.006.2018_ELA2_Year2_20180205.eaf"
    # Get the action at every 30 seconds of the video
    # act = get_actions_every_30s(annot_path, step_ms=30000)
    # timestamps = [time['timestamp_ms'] for time in act]
    # actions = [action['actions'] for action in act]
    # frame_annotations = get_eaf_labels(annot_path, None)
    # timestamps = sorted(frame_annotations.keys())[::(1800*30)][:6]  # default to first 100 if many
    # label_embeddings = create_label_embeddings(flo_model, processor, './csv/label_definitions.csv')
    # targets = list(label_embeddings.keys())

    # df = pd.read_csv('./csv/label_definitions.csv', header=None, names=['label', 'definition'], engine='python')
    # gt_labels = [(row['label'], row['definition']) for index, row in df.iterrows() if index!=0]

    # debug = False
    # images = get_images(video_path, timestamps)
    # labels = [[annot.strip() for annot in frame_annotations[key]] for key in timestamps]
    # if debug:
    #     # Create ground truth for each frame
    #     label_tensor = generate_ground_truth(labels, gt_labels)
        
    #     img_out_mean, lab_out_mean = create_img_and_label_embeddings(flo_model, processor, prompt, images, labels, gt_labels)
    #     _, matrix = cross_similarity(img_out_mean, lab_out_mean)
    #     bce = loss_ce(matrix, label_tensor.to(device=matrix.device, dtype=torch.float))
    
    vid_dir = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Videos_314 Hours/"
    annot_dir = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Video Annotations_314 Hours/"
    prompt = '<MORE_DETAILED_CAPTION>'
    batch_size = 8
    workers = 16

    print('Load file dictionary...')
    file_dict = load_file_dict(vid_dir, annot_dir)
    print('Set up education dataset and test...')
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

    def _optimizer(config, model, lambdas=[]):
        vision_params = list(map(id, model.module.vision_tower.parameters()))
        text_params = filter(lambda p: id(p) not in vision_params,
                             model.parameters())
        # Train just the text parameters
        optimizer = optim.SGD(text_params,
                              1e-5,
                              momentum=0.9,
                              weight_decay=0.2)
        for param in model.module.vision_tower.parameters():
            param.is_trainable = False
        return optimizer

    optimizer = _optimizer(None, flo_model)

    print(f'There are {len(train_loader)*batch_size} samples')
    print(f'There are {len(test_loader)*batch_size} samples')
    train(train_loader, test_loader, flo_model, processor, optimizer, prompt, edu.gt_labels, loss_info, loss_ce)

if __name__ == '__main__':
    main()