import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import logging

class EarlyStopper:
    def __init__(self, cfg):
        self.patience = cfg.training.patience
        self.min_delta = cfg.training.min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def compute_similarity(query_features, gallery_features):
    similarity_matrix = torch.matmul(query_features, gallery_features.T)
    return similarity_matrix

def evaluate_rank(
    similarity_matrix,
    query_ids,
    gallery_ids,
    query_c_ids,
    gallery_c_ids,
    max_rank=50
):

    num_queries = similarity_matrix.shape[0]

    all_cmc = []
    all_AP = []

    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_c_ids = np.asarray(query_c_ids)
    gallery_c_ids = np.asarray(gallery_c_ids)

    for q_idx in range(num_queries):
        q_id = query_ids[q_idx]
        q_cam = query_c_ids[q_idx]
        sims = similarity_matrix[q_idx]

        # sort all gallery images
        order = torch.argsort(sims, descending=True).cpu().numpy()
        
        g_id_ordered = gallery_ids[order]
        g_cam_ordered = gallery_c_ids[order]

        # remove same-id same-camera (junk images)
        valid_mask = ~((g_id_ordered == q_id) & (g_cam_ordered == q_cam))
        g_id_valid = g_id_ordered[valid_mask]

        matches = (g_id_valid == q_id).astype(np.int32)

        # no valid ground truth
        if matches.sum() == 0:
            continue

        # =========================
        # CMC
        # =========================
        first_match_idx = np.where(matches == 1)[0][0]
        cmc = np.zeros(max_rank)
        if first_match_idx < max_rank:
            cmc[first_match_idx:] = 1
        all_cmc.append(cmc)

        # =========================
        # Average Precision
        # =========================
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        
        match_indices = np.where(matches == 1)[0]
        precisions = tmp_cmc[match_indices] / (match_indices + 1.0)
        
        AP = np.sum(precisions) / num_rel
        all_AP.append(AP)

    # =========================
    # Final Metrics
    # =========================
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    mean_cmc = all_cmc.mean(axis=0) # numpy array shape (max_rank,)
    mAP = np.mean(all_AP) # float

    return mean_cmc, mAP

def evaluate(model, query_dataloader, test_dataloader, device):
    q_feat, q_ids, q_cams = extract_features(
        model,
        query_dataloader,
        device
    )

    g_feat, g_ids, g_cams = extract_features(
        model,
        test_dataloader,
        device
    )

    similarity_matrix = torch.matmul(
        q_feat,
        g_feat.t()
    )

    cmc, mAP = evaluate_rank(
        similarity_matrix,
        q_ids,
        g_ids,
        q_cams,
        g_cams
    )

    return cmc, mAP

def extract_features(model, dataloader, device):
    model.eval()

    features = []
    labels = []
    camera_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['img'].to(device)
            v_ids = batch['v_id']
            c_ids = batch['c_id']

            embedding = model(images)

            # L2 normalization
            embedding = nn.functional.normalize(embedding, p=2, dim=1)

            features.append(embedding.cpu())

            labels.extend(v_ids)
            camera_ids.extend(c_ids)

    features = torch.cat(features, dim=0)

    return features, labels, camera_ids

def extract_features_(model, image, device):
    model.eval()

    image = image.to(device)
    embedding = model(image)
    embedding = nn.functional.normalize(embedding, p=2, dim=1)

    feature = embedding.cpu()
    
    return feature

def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def save_checkpoint(state, save_dir, is_best):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    epoch = state['epoch']
    filename = os.path.join(save_dir, f'model_{epoch:03d}.pth')
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_filename)

def save_history(history, cfg):
    history_path = os.path.join(cfg.save_dir, 'history.json')

    safe_history = {}
    for key, value in history.items():
        safe_history[key] = [float(x) if isinstance(x, (np.floating, np.integer, float, int)) else x for x in value]
            
    with open(history_path, 'w') as f:
        json.dump(safe_history, f, indent=4)
        
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        if len(history['epochs']) > 0 and len(history['train_loss']) > 0:
            ax1.plot(history['epochs'], history['train_loss'], label='Train Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True)
        
        # Plot metrics
        if len(history['eval_epochs']) > 0:
            markers = ['o', 'x', '^', '*']
            for i, metric in enumerate(cfg.eval.metrics):
                if metric in history and len(history[metric]) > 0:
                    marker = markers[i % len(markers)]
                    ax2.plot(history['eval_epochs'], history[metric], label=metric, marker=marker)
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Score')
            ax2.set_title('Evaluation Metrics')
            ax2.legend()
            ax2.grid(True)
            
        plt.tight_layout()
        
        fig_name = getattr(cfg.plot, 'fig_name', 'metrics.png')
        fig_path = os.path.join(cfg.save_dir, fig_name)
        plt.savefig(fig_path)
        plt.close()
    except ImportError:
        pass

def get_image_paths(dataset):
    """
    Constructs the image paths directly from the dataset's samples list.
    """
    paths = []
    for _, _, _, img_path in dataset.samples:
        paths.append(img_path)
    return paths