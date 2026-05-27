import os
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import get_config
from load_dataset import VeRiDataset, VRICDataset, ValTransform
from model import Baseline, BaselineWithBOT, BaselineWithCBAM, BOTWithCBAM
from utils import extract_features, compute_similarity, evaluate_rank

def get_image_paths(dataset):
    """
    Constructs the image paths directly from the dataset's samples list.
    """
    paths = []
    for _, _, _, img_path in dataset.samples:
        paths.append(img_path)
    return paths

def main():
    parser = argparse.ArgumentParser(description="Test and visualize the best model")
    parser.add_argument('--dataset-name', type=str, default=None, help='Dataset name [VeRi776, VRIC]')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the model weights')
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device(cfg.system.device)
    
    if args.dataset_name is not None:
        cfg.dataset.name = args.dataset_name
        
    dataset_cfg = cfg.dataset.vric if cfg.dataset.name == 'VRIC' else cfg.dataset.veri
    
    # Auto-detect latest run if no argument
    best_model_path = args.model_path
    if not best_model_path:
        if os.path.exists(cfg.output_dir):
            runs = sorted([d for d in os.listdir(cfg.output_dir) if os.path.isdir(os.path.join(cfg.output_dir, d))])
            if len(runs) > 0:
                latest_run = runs[-1]
                cfg.save_dir = os.path.join(cfg.output_dir, latest_run)
                cfg.checkpoint.save_dir = os.path.join(cfg.save_dir, "checkpoint")
                best_model_path = os.path.join(cfg.checkpoint.save_dir, "best_model.pth")
                print(f"Auto-detected latest run: {latest_run}")
        
    if not best_model_path or not os.path.exists(best_model_path):
        print(f"Model not found at '{best_model_path}'. Please specify a valid --model_path.")
        return
        
    test_result_dir = os.path.join(os.path.dirname(os.path.dirname(best_model_path)), "test_result")
    os.makedirs(test_result_dir, exist_ok=True)

    # Data loaders
    val_transform = ValTransform(cfg)
    
    if cfg.dataset.name == 'VRIC':
        query_dataset = VRICDataset(dataset_cfg.query_dir, dataset_cfg.query_list, transform=val_transform)
        gallery_dataset = VRICDataset(dataset_cfg.gallery_dir, dataset_cfg.gallery_list, transform=val_transform)
    else:
        query_dataset = VeRiDataset(dataset_cfg.query_dir, transform=val_transform)
        gallery_dataset = VeRiDataset(dataset_cfg.gallery_dir, transform=val_transform)
    
    query_loader = DataLoader(
        query_dataset, batch_size=cfg.training.batch_size, shuffle=False, 
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=cfg.training.batch_size, shuffle=False, 
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory
    )
    
    query_img_paths = get_image_paths(query_dataset)
    gallery_img_paths = get_image_paths(gallery_dataset)
    
    if cfg.model.type == 'baseline':
        model = Baseline(cfg).to(device)
    elif cfg.model.type == 'baseline_with_BOT':
        model = BaselineWithBOT(cfg).to(device)
    elif cfg.model.type == 'baseline_with_CBAM':
        model = BaselineWithCBAM(cfg).to(device)
    elif cfg.model.type == 'BOT_with_CBAM':
        model = BOTWithCBAM(cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    
    print(f"Loading best model from {best_model_path}")
    loaded_dict = torch.load(best_model_path, weights_only=False, map_location=device)
    if 'state_dict' in loaded_dict:
        state_dict = loaded_dict['state_dict']
    else:
        state_dict = loaded_dict
        
    model_state_dict = model.state_dict()
    for k in list(state_dict.keys()): # remove classifier weights because of mismatch num_classes (e.g. trained on VeRi776 with 776 classes, but test on VRIC with 2811 classes)
        if k in model_state_dict and state_dict[k].shape != model_state_dict[k].shape:
            print(f"Skipping weight {k} due to shape mismatch (saved: {state_dict[k].shape}, current: {model_state_dict[k].shape})")
            del state_dict[k]
            
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Extraction
    print("Extracting gallery features...")
    gallery_features, gallery_v_ids, gallery_c_ids = extract_features(model, gallery_loader, device)
    
    print("Extracting query features...")
    query_features, query_v_ids, query_c_ids = extract_features(model, query_loader, device)
    
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity(query_features, gallery_features)
    
    # ======
    # Evaluate mAP and CMC
    # ======
    print("Evaluating metrics...")
    cmc, mAP = evaluate_rank(
        similarity_matrix,
        query_v_ids,
        gallery_v_ids,
        query_c_ids,
        gallery_c_ids
    )
    print(f"Test Results:")
    for metric in cfg.eval.metrics:
        if metric == 'mAP':
            print(f"mAP: {mAP:.4f}")
        elif metric.startswith('Rank-'):
            rank = int(metric.split('-')[1])
            print(f"Rank-{rank}: {cmc[rank-1]:.4f}")
    
    # ======
    # Test 1: Random samples
    # ======
    n_samples = 3
    n_top = 10
    
    random_indices = random.sample(range(len(query_dataset)), n_samples)
    fig, ax = plt.subplots(n_samples, n_top + 1, figsize=(20, 4 * n_samples))
    
    for i, q_idx in enumerate(random_indices):
        sim_score = similarity_matrix[q_idx]
        
        top_sim_score, top_index = torch.sort(sim_score, descending=True)
        top_sim_score = top_sim_score[:n_top]
        top_index = top_index[:n_top]
        
        retrieval_v_ids = [gallery_v_ids[idx] for idx in top_index]
        retrieval_img_paths = [gallery_img_paths[idx] for idx in top_index]
        q_v_id = query_v_ids[q_idx]
        
        ax[i][0].imshow(plt.imread(query_img_paths[q_idx]))
        ax[i][0].set_title(f"Query ID: {q_v_id}")
        ax[i][0].axis("off")
        
        for j in range(n_top):
            color = "green" if q_v_id == retrieval_v_ids[j] else "red"
            ax[i][j + 1].imshow(plt.imread(retrieval_img_paths[j]))
            ax[i][j + 1].set_title(f"Rank {j + 1}\nScore: {top_sim_score[j]:.3f}\nID: {retrieval_v_ids[j]}", color=color)
            for spine in ax[i][j + 1].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
            ax[i][j + 1].set_xticks([])
            ax[i][j + 1].set_yticks([])
            
    plt.tight_layout()
    
    vis1_path = os.path.join(test_result_dir, "random_retrieval_samples.png")
    os.makedirs(os.path.dirname(vis1_path), exist_ok=True)
    plt.savefig(vis1_path)
    plt.close()
    print(f"Saved random retrieval samples to {vis1_path}")
    
    # ======
    # Test 2: Hard samples (misses)
    # Samples that have at least one non-match in top n_top 
    # ======
    top_sim_score_all, top_index_all = torch.topk(similarity_matrix, k=n_top, dim=1)
    
    retrieval_miss_indices = []
    
    for i in range(len(query_v_ids)):
        q_id = query_v_ids[i]
        success = True
        
        for j in range(n_top):
            g_idx = top_index_all[i, j].item()
            g_id = gallery_v_ids[g_idx]
            
            if q_id != g_id:
                success = False
                
        if not success:
            retrieval_miss_indices.append(i)
            
    print(f"Number of retrieval misses (no match in top 10): {len(retrieval_miss_indices)}")
    
    n_show = min(5, len(retrieval_miss_indices))
    
    if n_show > 0:
        fig, ax = plt.subplots(n_show, n_top + 1, figsize=(20, 4 * n_show))
        fig.suptitle(f"Number of retrieval misses (no match in top 10): {len(retrieval_miss_indices)}", fontsize=16)
        if n_show == 1:
            ax = [ax]
            
        for row, q_idx in enumerate(retrieval_miss_indices[:n_show]):
            ax[row][0].imshow(plt.imread(query_img_paths[q_idx]))
            ax[row][0].set_title(f"Query\nID: {query_v_ids[q_idx]}")
            ax[row][0].axis("off")
            
            for j in range(n_top):
                g_idx = top_index_all[q_idx, j].item()
                g_id = gallery_v_ids[g_idx]
                color = "green" if g_id == query_v_ids[q_idx] else "red"
                
                ax[row][j + 1].imshow(plt.imread(gallery_img_paths[g_idx]))
                ax[row][j + 1].set_title(f"Rank {j+1}\nID: {g_id}\n{top_sim_score_all[q_idx, j]:.3f}", color=color)
                for spine in ax[row][j + 1].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(4)
                ax[row][j + 1].set_xticks([])
                ax[row][j + 1].set_yticks([])
                
        plt.tight_layout()
        misses_path = os.path.join(test_result_dir, "retrieval_misses.png")
        plt.savefig(misses_path)
        plt.close()
        print(f"Saved retrieval misses to {misses_path}")

if __name__ == '__main__':
    main()
