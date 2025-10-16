# TODO
# RQ
#   After training use flo_model.generate and see how the text output looks
#   Compare MORE_DETAILED_CAPTION, DETAILED_CAPTION, and CAPTION
#   How are the bounding boxes for the BB prompt?
#   How does training <CAPTION> effect <DETAILED_CAPTION> or <MORE_DETAILED_CAPTION> and vice versa
# TO TEST
#   AdamW vs SGD
#   Loss function infoNCE vs CE vs both (balancing them)
#   Learning rates and scheduling learning rates
#   Size of input image (224x224) vs (512x512) vs (768x768)

from src.similarities import cross_similarity
from src.create_embeddings import create_embeddings as create_label_embeddings
from src.visualize import visualize_cross_similarity
from src.computations import get_captions, next_word_similarity, calculate_similarity, get_df
from src.timestamp_captions import get_timestamp_captions, create_caption_embeddings, get_images, extract_region_and_label_embeddings
from src.eaf_labels import get_eaf_labels
from aiai_dataloader.datasets import EducationDataset, load_file_dict
from thumos_dataloader import MultiTHUMOSDataset

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
import torch.nn.functional as F
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
from dotmap import DotMap
import argparse
import yaml

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
    for labels in gt_labels:
        # Process label
        if type(labels) == tuple:
            label, definition = labels
        else:
            label = labels
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

# def calculate_accuracy(matrix, labels):
#     num = 0
#     corr_k = 0
#     labeled_ids = []
#     correct_ids = []
#     tp = torch.zeros(labels.shape[-1], device=matrix.device)
#     tn = torch.zeros(labels.shape[-1], device=matrix.device)
#     fp = torch.zeros(labels.shape[-1], device=matrix.device)
#     fb = torch.zeros(labels.shape[-1], device=matrix.device)
#     assert matrix.shape[0] == labels.shape[0]
#     num_batches, d = matrix.shape
#     _, num_labels = labels.shape
#     for i in range(num_batches):
#         mat = matrix[i]
#         label_list = labels[i]
#         mat_list = ((matrix/torch.sum(matrix, dim=1).unsqueeze(dim=1)) > (1/labels.shape[-1] + .01)).to(device='cpu', dtype=torch.float)
#         # mat_list = [1.0 if x > 1/len(label_list) else 0.0 for x in mat.tolist()]
#         for pred, label in zip(mat_list, label_list):
#             if label == pred:
#                 if label == 1:
#                     tp += 1
#                 elif label == 0:
#                     tn += 1
#                 corr_k += 1
#             num += 1
#         # find the indices where the gt is truly 1
#         indices = [i for i, x in enumerate(label_list) if x == 1]
#         # Get the top k guesses where k is the length of gt
#         values_k, indices_k = mat.topk(len(indices), dim=-1)
#         labeled_ids.append(indices_k.tolist())
#         correct_ids.append(indices)
        
    return num, corr_k, tp, tn, labeled_ids, correct_ids

# https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
def get_conf_components(matrix, labels, threshold, threshold_method='sigmoid', return_corr=False):
    if threshold_method == 'sigmoid':
        mat = (F.sigmoid(matrix) > 0.5).to(dtype=torch.float)
    elif threshold_method == 'average':
        mat = ((matrix/torch.sum(matrix, dim=1).unsqueeze(dim=1)) > (1/labels.shape[-1] + .01)).to(dtype=torch.float)
    elif threshold_method == 'khot':
        indices = [[i for i, x in enumerate(label_list) if x == 1] for label_list in labels]
        indices_k = [matrix[i].topk(len(index_list), dim=-1)[-1] for i, index_list in enumerate(indices)]
        pred = []
        for index_k in indices_k:
            pred.append(torch.tensor([1 if i in index_k else 0 for i in range(labels.shape[-1])]))
        mat = torch.stack(pred).to(device=labels.device)
        # mat = ((matrix/torch.sum(matrix, dim=1).unsqueeze(dim=1)) > (1/labels.shape[-1] + .01)).to(dtype=torch.float)
    elif threshold_method == 'softmax':
        mat = (F.softmax(matrix) > 0.5).to(dtype=torch.float)
    else:
        raise ValueError(f'No threshold method called {threshold_method}')

    check      = (mat == labels)
    yhat       = labels.to(dtype=torch.bool)
    not_check  = torch.logical_not(check)
    not_yhat   = torch.logical_not(yhat)

    tp = torch.sum(torch.logical_and(yhat,     check).to(dtype=int), dim=0)
    tn = torch.sum(torch.logical_and(not_yhat, check).to(dtype=int), dim=0)
    fn = torch.sum(torch.logical_and(yhat,     not_check).to(dtype=int), dim=0)
    fp = torch.sum(torch.logical_and(not_yhat, not_check).to(dtype=int), dim=0)

    corr_k = torch.sum(check.to(dtype=int))
    num    = check.shape[0]*check.shape[1]

    if return_corr:
        assert corr_k == torch.sum(tp + tn)
        assert num == torch.sum(tp + tn + fp + fn)

        return num, corr_k, (tp, tn, fp, fn), mat
    else:
        return tp, tn, fp, fn, mat

def get_metrics(tp, tn, fp, fn):
    p = tp/(tp+fp+1e-12)
    r = tp/(tp+fn+1e-12)

    #unique_r, inverse_indices = torch.unique(r, return_inverse=True)
    #max_p_per_group = torch.zeros_like(unique_r)
    #max_p_per_group = max_p_per_group.scatter_reduce(0, inverse_indices, p, reduce='amax', include_self=False)
    #p_new = max_p_per_group[inverse_indices]

    #idx = torch.arange(len(r))
    #last_occurrence = scatter_last = torch.zeros_like(r, dtype=torch.bool)
    #scatter_last = scatter_last.to(dtype=inverse_indices.dtype).scatter(0, inverse_indices, idx.to(dtype=inverse_indices.dtype))
    #mask = (idx == scatter_last[inverse_indices])

    #r_new = r.clone()
    #r_new[~mask] = 0
    #keep_mask = mask.int()

    ap = p*r #torch.sum(p_new*keep_mask)/torch.sum(keep_mask)
    map = ap.mean()
    f1 = ((2*p*r)/(p+r+1e-12)).mean()
    return f1, map

def khot_accuracy(matrix, labels, output_ids=False):
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

    if output_ids:
        return num, corr_k, labeled_ids, correct_ids
    else:
        return num, corr_k

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

def get_loss(loss_ce, loss_img_to_txt, img_to_txt_logits, img_out_mean, lab_out_mean, labels, config):
    labels = labels.to(device=img_to_txt_logits.device, dtype=torch.float)
    ground_truth = gen_multi_label(labels, img_to_txt_logits.device)
    ce_loss = loss_ce(img_to_txt_logits, labels)
    info_loss = contrastive_info_nce(batch_embs=img_out_mean, label_embs=lab_out_mean, targets=labels, temperature=0.07)
    img_to_txt = loss_img_to_txt(img_to_txt_logits, ground_truth)
    return config.loss.ce*ce_loss + config.loss.info*info_loss + config.loss.i2t*img_to_txt

def validate(test_loader, flo_model, processor, prompt, gt_labels, epoch, epochs, loss_img_to_txt, loss_ce, cfg, threshold, debug=False):
    flo_model.eval()
    total_num = 0
    total_corr = 0
    if cfg.debug:
        model = flo_model
    else:
        model = flo_model.module
    tp_all = torch.zeros(len(gt_labels), device=model.device)
    tn_all = torch.zeros(len(gt_labels), device=model.device)
    fp_all = torch.zeros(len(gt_labels), device=model.device)
    fn_all = torch.zeros(len(gt_labels), device=model.device)
    total_labeled_ids = []
    total_correct_ids = []
    train_loss = 0
    calc_tp = False
    with torch.no_grad():
        for (images, timestamp, labels) in tqdm(test_loader, total=len(test_loader)):
            img_out_mean, lab_out_mean = create_img_and_label_embeddings(flo_model, processor, prompt, images, gt_labels, cfg, debug=debug)
            _, img_to_txt_logits = cross_similarity(img_out_mean, lab_out_mean)
            # img_to_txt_logits = torch.nn.functional.sigmoid(img_to_txt_logits)
            loss = get_loss(loss_ce, loss_img_to_txt, img_to_txt_logits, img_out_mean, lab_out_mean, labels, cfg)
            labels = labels.to(device=img_to_txt_logits.device)
            if cfg.option == '1':
                num, corr_k, (tp, tn, fp, fn), khot_pred = get_conf_components(img_to_txt_logits, labels, threshold, return_corr=True)
                
                tp_all += tp
                tn_all += tn
                fp_all += fp
                fn_all += fn
            elif cfg.option == '2':
                tp, tn, fp, fn, khot_pred = get_conf_components(img_to_txt_logits, labels, threshold, threshold_method='khot')
                tp_all += tp
                tn_all += tn
                fp_all += fp
                fn_all += fn
                num, corr_k, labeled_ids, correct_ids = khot_accuracy(img_to_txt_logits, labels, output_ids=True)
            total_num += num
            total_corr += corr_k

            train_loss += loss.item()

            # Two lists of lists where each element of the list is a list of predicted labels and a list of correct labels
            total_labeled_ids.extend(khot_pred.tolist())
            total_correct_ids.extend(labels.tolist())

    f1, mAP = get_metrics(tp_all, tn_all, fp_all, fn_all)
    print(f'Epoch: [{epoch+1}/{epochs}]: Test Loss: {train_loss/len(test_loader)}, Top1: {total_corr}/{total_num} = {(total_corr/total_num)*100:.2f}%, F1: {f1:.2f}, mAP: {mAP:.2f}')

    if cfg.print:
        plot_confusion_matrix(total_correct_ids, total_labeled_ids, np.array(gt_labels), cfg.name)

    return total_corr/total_num

def train(train_loader, test_loader, flo_model, processor, optimizer, prompt, gt_labels, loss_img_to_txt, loss_ce, lr_scheduler, cfg, threshold, debug=False):
    best_acc = 0
    for epoch in range(cfg.solver.epochs):
        debug_count = 0
        train_loss = 0
        flo_model.train()
        total_num = 0
        total_corr = 0
        if cfg.debug:
            model = flo_model
        else:
            model = flo_model.module
        tp_all = torch.zeros(len(gt_labels), device=model.device)
        tn_all = torch.zeros(len(gt_labels), device=model.device)
        fn_all = torch.zeros(len(gt_labels), device=model.device)
        fp_all = torch.zeros(len(gt_labels), device=model.device)
        for kkk, (images, timestamp, labels) in enumerate(tqdm(train_loader, total=len(train_loader))):
            if cfg.solver.schedule and ((kkk+1) == 1 or (kkk+1) % 10 == 0):
                lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            img_out_mean, lab_out_mean = create_img_and_label_embeddings(flo_model, processor, prompt, images, gt_labels, cfg, debug=debug)
            _, img_to_txt_logits = cross_similarity(img_out_mean, lab_out_mean)
            # img_to_txt_logits = torch.nn.functional.sigmoid(img_to_txt_logits)

            # _, txt_to_img_logits = cross_similarity(lab_out_mean, img_out_mean)
            labels = labels.to(device=img_to_txt_logits.device, dtype=torch.float)

            if cfg.option == '1':
                num, corr_k, (tp, tn, fp, fn), khot_pred = get_conf_components(img_to_txt_logits, labels, threshold, return_corr=True)
                tp_all += tp
                tn_all += tn
                fn_all += fn
                fp_all += fp
            elif cfg.option == '2':
                tp, tn, fp, fn, khot_pred = get_conf_components(img_to_txt_logits, labels, threshold, threshold_method='khot')
                tp_all += tp
                tn_all += tn
                fn_all += fn
                fp_all += fp
                num, corr_k = khot_accuracy(img_to_txt_logits, labels)
            total_num += num
            total_corr += corr_k

            loss = get_loss(loss_ce, loss_img_to_txt, img_to_txt_logits, img_out_mean, lab_out_mean, labels, cfg)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        f1, mAP = get_metrics(tp_all, tn_all, fp_all, fn_all)
        print(f"Average Training Loss: {avg_train_loss}, Accuracy: {total_corr}/{total_num} = {(total_corr/total_num)*100:.2f}%, F1: {f1:.2f}, mAP: {mAP:.2f}")
        
        if (epoch+1) % cfg.freq == 0:
            acc = validate(test_loader, flo_model, processor, prompt, gt_labels, epoch, cfg.solver.epochs, loss_img_to_txt, loss_ce, cfg, threshold, debug=debug)

            # TODO add best model analysis and saving here
            if acc > best_acc:
                if cfg.debug:
                    model = flo_model
                else:
                    model = flo_model.module
                best_acc = acc
                print(f'Best accuracy is {best_acc*100:.2f}%')
                # save model here
                model.save_pretrained(cfg.output_dir)
                processor.save_pretrained(cfg.output_dir)

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
        optimizer = optim.AdamW(text_params, betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8, weight_decay=.2)
    elif mode == 'sgd':
        optimizer = optim.SGD(text_params, config.solver.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    total_params = sum(p.numel() for p in flo_model.parameters())
    trainable_params = sum(p.numel() for p in flo_model.parameters() if p.requires_grad)
    print(f'PARAMS: {total_params}, Trainable: {trainable_params}')
    return optimizer

def _lr_scheduler(cfg,optimizer):
    if cfg.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            cfg.solver.epochs,
            warmup_epochs=cfg.solver.lr_warmup_step
        )
    elif cfg.solver.type == 'multistep':
        if isinstance(cfg.solver.lr_decay_step, list):
            milestones = cfg.solver.lr_decay_step
        elif isinstance(cfg.solver.lr_decay_step, int):
            milestones = [
                cfg.solver.lr_decay_step * (i + 1)
                for i in range(cfg.solver.epochs //
                               cfg.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(cfg.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(cfg.solver.type))
    return lr_scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    # parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # set up loss functions
    loss_ce = torch.nn.BCEWithLogitsLoss()
    loss_img_to_txt = KLLoss()
    loss_txt_to_img = KLLoss()

    cfg = DotMap(config)

    if cfg.seed is not None:
        seed = int(cfg.seed)
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)  # For GPU operations
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
        random.seed(seed)
        np.random.seed(seed) 
    else:
        seed = int(random.random()*10e7)
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)  # For GPU operations
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
        random.seed(seed)
        np.random.seed(seed) 
        cfg.seed = seed
    print(f'SEED: {seed}')

    if cfg.dataset == 'aiai':
        print('Load file dictionary...')
        file_dict, desc_list = load_file_dict(cfg.vid_dir, cfg.annot_dir, stunt=cfg.stunt)

        print('Set up education dataset and test...')

        # determine the train-test split
        indices = np.arange(len(desc_list))
        np.random.shuffle(indices)
        split_idx = int(.8 * len(indices))
        train_desc_list = [desc_list[ind] for ind in indices[:split_idx]]
        test_desc_list = [desc_list[ind] for ind in indices[split_idx:]]

        dataset_train = EducationDataset(cfg.vid_dir, cfg.annot_dir, file_dict, train_desc_list, transform=transforms.ToTensor(), size=cfg.size, label_defs=cfg.label_defs)
        dataset_test = EducationDataset(cfg.vid_dir, cfg.annot_dir, file_dict, test_desc_list, transform=transforms.ToTensor(), size=cfg.size, label_defs=cfg.label_defs)
        print(f'There are {len(list(dataset_train.get_file_dict().keys()))} video-eaf pairs')
        all_labels = dataset_train.gt_labels
    elif cfg.dataset == 'multithumos':
        dataset_train = MultiTHUMOSDataset(dataset_path=cfg.vid_dir, annotations=cfg.annot_dir, stunt=.05, transform=transforms.ToTensor(), train=True, label_csv=cfg.label_defs)
        dataset_test  = MultiTHUMOSDataset(dataset_path=cfg.vid_dir, annotations=cfg.annot_dir, stunt=.02, transform=transforms.ToTensor(), train=False, label_csv=cfg.label_defs)
        all_labels = dataset_train.labels
    else:
        raise ValueError(f'Unknwon dataset {cfg.dataset}')

    def collate_fn(batch):
        image, timestamp, label = zip(*batch)
        # Check the labels for bb
        image = torch.stack(image)
        timestamps = torch.tensor(timestamp)
        labels = pad_sequence(label, batch_first=True, padding_value=-1)
        return image, timestamp, labels

    train_loader = DataLoader( dataset_train, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=True, pin_memory=False, drop_last=True,  collate_fn=collate_fn)
    test_loader = DataLoader( dataset_test, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=False, pin_memory=False, drop_last=True,  collate_fn=collate_fn)
    # import pdb; pdb.set_trace()

    print(f'There are {len(train_loader)*cfg.batch_size} samples')
    print(f'There are {len(test_loader)*cfg.batch_size} samples')

    flo_model, processor = flor2.load("BASE_FT", device, lora=cfg.lora)
    if cfg.debug:
        flo_model = flo_model.to(device)
    else:
        flo_model = torch.nn.DataParallel(flo_model).cuda()

    optimizer = _optimizer(cfg, flo_model, debug=cfg.debug, mode=cfg.solver.mode)
    lr_scheduler = _lr_scheduler(cfg, optimizer)

    threshold = (1/len(all_labels)) + .01
    validate(test_loader, flo_model, processor, cfg.prompt, all_labels, -1, cfg.solver.epochs, loss_img_to_txt, loss_ce, cfg, threshold, debug=cfg.debug)

    train(train_loader, test_loader, flo_model, processor, optimizer, cfg.prompt, all_labels, loss_img_to_txt, loss_ce, lr_scheduler, cfg, threshold, debug=cfg.debug)

if __name__ == '__main__':
    main()
