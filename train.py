# TODO
# RQ
#   After training use flo_model.generate and see how the text output looks
#   Compare MORE_DETAILED_CAPTION, DETAILED_CAPTION, and CAPTION
#   How are the bounding boxes for the BB prompt?
# TO TEST
#   AdamW vs SGD
#   Loss function infoNCE vs CE vs both (balancing them)
#   Learning rates and scheduling learning rates
#   Size of input image (224x224) vs (512x512)

from src.similarities import cross_similarity
from src.create_embeddings import create_embeddings as create_label_embeddings
from src.visualize import visualize_cross_similarity
from src.computations import get_captions, next_word_similarity, calculate_similarity, get_df
from src.timestamp_captions import get_timestamp_captions, create_caption_embeddings, get_images, extract_region_and_label_embeddings
from src.eaf_labels import get_eaf_labels
from datasets import EducationDataset, load_file_dict

import florence_pytorch.florence.modeling_florence2 as flor2
import torch
import random
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
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR

# https://medium.com/@elsayed_mohamed/florence-2-vlm-fine-tuning-on-custom-dataset-3dd231585091
from transformers import get_scheduler
from utils.KLLoss import KLLoss, gen_multi_label

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

def create_embeddings(flo_model, processor, gt_labels, debug=False):
    embedding_data = []
    attention_mask = []
    if not debug:
        flo_model = flo_model.module
    for label, definition in gt_labels:
        # Process label
        tokens = processor.tokenizer(label)
        embedding_data.append(torch.tensor(tokens['input_ids'], device=flo_model.device))
        attention_mask.append(torch.tensor(tokens['attention_mask'], device=flo_model.device))

    input_ids = pad_with_preserved_tokens(embedding_data)
    inputs_embeds = torch.stack([flo_model.get_input_embeddings()(tensor) for tensor in input_ids])
    # 0 to ignore these masked tokens
    attention_mask = pad_with_preserved_tokens(attention_mask, pad_value=0)
    
    return input_ids, inputs_embeds, attention_mask
    
def create_img_and_label_embeddings(flo_model, processor, prompt, sample, gt_labels, cfg, debug=False):
    prompts = [prompt for _ in sample]
    samples = [img for img in sample]
    
    # Create prompt/sample inputs
    if not debug:
        model = flo_model.module
    else:
        model = flo_model
    # import pdb; pdb.set_trace()
    processor.image_processor.do_rescale = False
    processor.image_processor.size = cfg.size
    inputs = processor(text=prompts, images=samples, return_tensors="pt", padding=True, do_rescale=False)
    input_ids = inputs['input_ids'].to(device=model.device)
    pixel_values = inputs['pixel_values'].to(device=model.device)
    decoder_img_ids = flor2.shift_tokens_right(input_ids.to(dtype=int), model.config.pad_token_id, 0).to(device=model.device, dtype=int)
    img_out   = flo_model(input_ids = input_ids,
                        pixel_values = pixel_values,
                        decoder_input_ids = decoder_img_ids,
                        output_hidden_states = True)

    # Create all label embeddings in a list ordered by label name
    label_embeddings, label_embeddings_embeds, attention_mask = create_embeddings(flo_model, processor, gt_labels, debug=debug)
    
    decoder_label_ids = flor2.shift_tokens_right(label_embeddings.to(dtype=int), model.config.pad_token_id, 0).to(device=model.device, dtype=int)
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
    tp = 0
    tn = 0
    for mat, label_list in zip(matrix, labels):
        mat_list = ((matrix/torch.sum(matrix, dim=1).unsqueeze(dim=1)) > (1/11 + .01)).to(device='cpu', dtype=torch.float)
        # mat_list = [1.0 if x > 1/len(label_list) else 0.0 for x in mat.tolist()]
        for pred, label in zip(mat_list, label_list):
            if label == pred:
                if label == 1:
                    tp += 1
                elif label == 0:
                    tn += 1
                corr_k += 1
            num += 1
        # find the indices where the gt is truly 1
        indices = [i for i, x in enumerate(label_list) if x == 1]
        # Get the top k guesses where k is the length of gt
        values_k, indices_k = mat.topk(len(indices), dim=-1)
        labeled_ids.append(indices_k.tolist())
        correct_ids.append(indices)
        
    return num, corr_k, tp, tn, labeled_ids, correct_ids

def calculate_accuracy_old(matrix, labels):
    num = 0
    corr_k = 0
    labeled_ids = []
    correct_ids = []
    for mat, label_list in zip(matrix, labels):
        # find the indices where the gt is truly 1
        indices = [i for i, x in enumerate(label_list) if x == 1]
        # Get the top k guesses where k is the length of gt
        values_k, indices_k = mat.topk(len(indices), dim=-1)
        for ind in indices:
            if ind in indices_k:
                corr_k += 1
            num += 1
        labeled_ids.append(indices_k.tolist())
        correct_ids.append(indices)
    return num, corr_k, labeled_ids, correct_ids

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

    cm = multilabel_confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    with open(f'confusion_{name}.txt', 'w') as f:
        f.write(classification_report(y_true,y_pred, target_names=[classname[0].replace('-', '_').replace(' ', '_') for classname in classes]))

def get_loss(loss_ce, loss_img_to_txt, img_to_txt_logits, img_out_mean, lab_out_mean, labels):
    labels = labels.to(device=img_to_txt_logits.device, dtype=torch.float)
    ground_truth = gen_multi_label(labels, img_to_txt_logits.device)
    ce_loss = loss_ce(img_to_txt_logits, labels)
    # info_loss = contrastive_info_nce(batch_embs=img_out_mean, label_embs=lab_out_mean, targets=labels, temperature=0.07)
    # img_to_txt = loss_img_to_txt(img_to_txt_logits, ground_truth)
    return ce_loss #.6ce_loss+ .4*info_loss

def validate(test_loader, flo_model, processor, prompt, gt_labels, epoch, epochs, loss_img_to_txt, loss_ce, cfg, debug=False):
    flo_model.eval()
    total_num = 0
    total_corr = 0
    total_labeled_ids = []
    total_correct_ids = []
    train_loss = 0
    with torch.no_grad():
        for (images, timestamp, labels) in tqdm(test_loader, total=len(test_loader)):
            img_out_mean, lab_out_mean = create_img_and_label_embeddings(flo_model, processor, prompt, images, gt_labels, cfg, debug=debug)
            _, img_to_txt_logits = cross_similarity(img_out_mean, lab_out_mean)
            loss = get_loss(loss_ce, loss_img_to_txt, img_to_txt_logits, img_out_mean, lab_out_mean, labels)
            if cfg.option == '1':
                num, corr_k, labeled_ids, correct_ids = calculate_accuracy_old(img_to_txt_logits, labels)
            elif cfg.option == '2':
                num, corr_k, tp, tn, labeled_ids, correct_ids = calculate_accuracy(img_to_txt_logits, labels)
            total_num += num
            total_corr += corr_k
            
            train_loss += loss.item()
                
            # Two lists of lists where each element of the list is a list of predicted labels and a list of correct labels
            total_labeled_ids.extend(((img_to_txt_logits/torch.sum(img_to_txt_logits, dim=1).unsqueeze(dim=1)) > (1/11 + .01)).to(device='cpu', dtype=torch.float))
            total_correct_ids.extend(labels.tolist())

    if cfg.print:
        plot_confusion_matrix(total_correct_ids, total_labeled_ids, np.array(gt_labels), cfg.name)
    
    print(f'Epoch: [{epoch+1}/{epochs}]: Test Loss: {train_loss/len(test_loader)}, Top1: {total_corr}/{total_num} = {(total_corr/total_num)*100}%')

def train(train_loader, test_loader, flo_model, processor, optimizer, prompt, gt_labels, loss_img_to_txt, loss_ce, lr_scheduler, cfg, debug=False):
    for epoch in range(cfg.solver.epochs):
        debug_count = 0
        train_loss = 0
        flo_model.train()
        for kkk, (images, timestamp, labels) in enumerate(tqdm(train_loader, total=len(train_loader))):
            # if cfg.schedule:
            #     if (kkk+1) == 1 or (kkk+1) % 10 == 0:
            #         lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            img_out_mean, lab_out_mean = create_img_and_label_embeddings(flo_model, processor, prompt, images, gt_labels, cfg, debug=debug)
            _, img_to_txt_logits = cross_similarity(img_out_mean, lab_out_mean)
            # _, txt_to_img_logits = cross_similarity(lab_out_mean, img_out_mean)
            labels = labels.to(device=img_to_txt_logits.device, dtype=torch.float)

            loss = get_loss(loss_ce, loss_img_to_txt, img_to_txt_logits, img_out_mean, lab_out_mean, labels)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")
        if (epoch+1) % cfg.freq == 0:
            validate(test_loader, flo_model, processor, prompt, gt_labels, epoch, cfg.solver.epochs, loss_img_to_txt, loss_ce, cfg, debug=debug)
    
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

def _optimizer(config, flo_model, debug=False, mode='adamw'):
    if not debug:
        flo_model = flo_model.module
    vision_params = list(map(id, flo_model.vision_tower.parameters()))
    # Freeze weights
    if config.freeze_fc:
        for layer in flo_model.language_model.model.decoder.layers:
            for param in layer.fc1.parameters():
                param.requires_grad = False
            for param in layer.fc2.parameters():
                param.requires_grad = False
        for layer in flo_model.language_model.model.encoder.layers:
            for param in layer.fc1.parameters():
                param.requires_grad = False
            for param in layer.fc2.parameters():
                param.requires_grad = False
    for param in flo_model.vision_tower.parameters():
        param.requires_grad = False
    # Train just the text parameters
    text_params = filter(lambda p: (id(p) not in vision_params) and p.requires_grad, flo_model.parameters())
    if mode =='adamw':
        optimizer = optim.AdamW(text_params,
                            betas=(0.9, 0.98), lr=config.lr, eps=1e-8,
                            weight_decay=.2)
    elif mode == 'sgd':
        optimizer = optim.SGD(text_params, config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    total_params = sum(p.numel() for p in flo_model.parameters())
    trainable_params = sum(p.numel() for p in flo_model.parameters() if p.requires_grad)
    print(f'PARAMS: {total_params}, Trainable: {trainable_params}')
    return optimizer

# def _lr_scheduler(cfg,optimizer):
#     solver = cfg.solver
#     if solver.type == 'cosine':
#         lr_scheduler = WarmupCosineAnnealingLR(
#             optimizer,
#             cfg.solver.epochs,
#             warmup_epochs=solver.lr_warmup_step
#         )
#     elif solver.type == 'multistep':
#         if isinstance(solver.lr_decay_step, list):
#             milestones = solver.lr_decay_step
#         elif isinstance(solver.lr_decay_step, int):
#             milestones = [
#                 solver.lr_decay_step * (i + 1)
#                 for i in range(cfg.solver.epochs //
#                                solver.lr_decay_step)]
#         else:
#             raise ValueError("error learning rate decay step: {}".format(type(solver.lr_decay_step)))
#         lr_scheduler = WarmupMultiStepLR(
#             optimizer,
#             milestones,
#             warmup_epochs=solver.lr_warmup_step
#         )
#     else:
#         raise ValueError('Unknown lr scheduler: {}'.format(solver.type))
#     return lr_scheduler

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 
    
    # set up loss functions
    loss_ce = torch.nn.BCEWithLogitsLoss()
    loss_img_to_txt = KLLoss()
    loss_txt_to_img = KLLoss()
    
    # For config file
    class Cfg():
        vid_dir = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Videos_314 Hours/"
        annot_dir = "/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Video Annotations_314 Hours/"
        prompt = '<MORE_DETAILED_CAPTION>'
        batch_size = 8
        workers = 8
        debug=False
        stunt=.05
        label_defs='/scratch/tkg5kq/sandbox/florence_experiments/csv/label_definitions_limited.csv'
        epochs=50
        lr=1e-3
        # momentum=0.9
        seed=None
        edu_seed=42
        mode='adamw'
        # weight_decay=0.2
        schedule=False
        ce = 1
        info=0
        option='1'
        name='test'
        freeze_fc=True
        lora=False
        size=512
        freq = 2
        print=False
        
        class Solver():
            epochs= 50
            # type= 'cosine'
            # lr_warmup_step= 5
            # lr_decay_step= 15
            
        solver = Solver()

    cfg = Cfg()
    
    flo_model, processor = flor2.load("BASE_FT", device, lora=cfg.lora)
    if cfg.debug:
        flo_model = flo_model.to(device)
    else:
        flo_model = torch.nn.DataParallel(flo_model).cuda()

    print('Load file dictionary...')
    file_dict = load_file_dict(cfg.vid_dir, cfg.annot_dir, stunt=cfg.stunt)

    print('Set up education dataset and test...')
    edu = EducationDataset(cfg.vid_dir, cfg.annot_dir, file_dict, transform=transforms.ToTensor(), train=True, size=cfg.size, label_defs=cfg.label_defs, seed=cfg.edu_seed)
    edu_test = EducationDataset(cfg.vid_dir, cfg.annot_dir, file_dict, transform=transforms.ToTensor(), train=False, size=cfg.size, label_defs=cfg.label_defs, seed=cfg.edu_seed)
    print(f'There are {len(list(edu.get_file_dict().keys()))} video-eaf pairs')
    
    # if cfg.seed is not None:
    #     seed = int(config['seed'])
    #     torch.manual_seed(seed) 
    #     torch.cuda.manual_seed(seed)  # For GPU operations
    #     torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    #     random.seed(seed)
    #     np.random.seed(seed) 
    # else:
    #     seed = int(random.random()*10e7)
    #     torch.manual_seed(seed) 
    #     torch.cuda.manual_seed(seed)  # For GPU operations
    #     torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    #     random.seed(seed)
    #     np.random.seed(seed) 
    #     cfg.seed = seed
    # print(f'SEED: {seed}')


    def collate_fn(batch):
        image, timestamp, label = zip(*batch)
        # Check the labels for bb
        image = torch.stack(image) 
        timestamps = torch.tensor(timestamp)
        labels = pad_sequence(label, batch_first=True, padding_value=-1)
        return image, timestamp, labels

    train_loader = DataLoader( edu, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=False, pin_memory=False, drop_last=True,  collate_fn=collate_fn)
    test_loader = DataLoader( edu_test, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=False, pin_memory=False, drop_last=True,  collate_fn=collate_fn)

    optimizer = _optimizer(cfg, flo_model, debug=cfg.debug, mode=cfg.mode)
    # lr_scheduler = _lr_scheduler(cfg, optimizer)

    print(f'There are {len(train_loader)*cfg.batch_size} samples')
    print(f'There are {len(test_loader)*cfg.batch_size} samples')

    validate(test_loader, flo_model, processor, cfg.prompt, edu.gt_labels, -1, cfg.solver.epochs, loss_img_to_txt, loss_ce, cfg, debug=cfg.debug)

    train(train_loader, test_loader, flo_model, processor, optimizer, cfg.prompt, edu.gt_labels, loss_img_to_txt, loss_ce, None, cfg, debug=cfg.debug)

if __name__ == '__main__':
    main()