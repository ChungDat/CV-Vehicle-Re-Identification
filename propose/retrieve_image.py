import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import get_config
from load_dataset import VeRiDataset, VRICDataset, ValTransform
from model import Baseline, BaselineWithBOT, BaselineWithCBAM, BOTWithCBAM
from utils import extract_features, compute_similarity, get_image_paths

def main():
    parser = argparse.ArgumentParser(description="Retrieve images given an input image name")
    parser.add_argument('--dataset-name', type=str, default=None, help='Dataset name [VeRi776, VRIC]')
    parser.add_argument('--model-type', type=str, default=None, help='Model type [baseline, baseline_with_BOT, baseline_with_CBAM, BOT_with_CBAM]')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--img-name', type=str, nargs='+', required=True, help='Name(s) of the input image(s) (e.g. 0002_c002_00030600_0.jpg for VeRi776, MVI_20011_002_img00010.jpg for VRIC)')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top retrieval results to show')
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device(cfg.system.device)
    
    if args.dataset_name is not None:
        cfg.dataset.name = args.dataset_name
    if args.model_type is not None:
        cfg.model.type = args.model_type
        
    dataset_cfg = cfg.dataset.vric if cfg.dataset.name == 'VRIC' else cfg.dataset.veri
    
    if not os.path.exists(args.model_path):
        print(f"Model not found at '{args.model_path}'.")
        return

    val_transform = ValTransform(cfg)
    
    if cfg.dataset.name == 'VRIC':
        query_dataset = VRICDataset(dataset_cfg.query_dir, dataset_cfg.query_list, transform=val_transform)
        gallery_dataset = VRICDataset(dataset_cfg.gallery_dir, dataset_cfg.gallery_list, transform=val_transform)
    else:
        query_dataset = VeRiDataset(dataset_cfg.query_dir, transform=val_transform)
        gallery_dataset = VeRiDataset(dataset_cfg.gallery_dir, transform=val_transform)
    
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=cfg.training.batch_size, shuffle=False, 
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory
    )
    
    query_img_paths = get_image_paths(query_dataset)
    gallery_img_paths = get_image_paths(gallery_dataset)
    
    # Load model
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
    
    print(f"Loading best model from {args.model_path}")
    loaded_dict = torch.load(args.model_path, weights_only=False, map_location=device)
    state_dict = loaded_dict.get('state_dict', loaded_dict)
        
    model_state_dict = model.state_dict()
    for k in list(state_dict.keys()): # remove classifier weights because of mismatch num_classes (e.g. trained on VeRi776 with 776 classes, but test on VRIC with 2811 classes)
        if k in model_state_dict and state_dict[k].shape != model_state_dict[k].shape:
            del state_dict[k]
            
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    missing_keys = [k for k in missing_keys if 'classifier' not in k]
    unexpected_keys = [k for k in unexpected_keys if 'classifier' not in k]
    
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print(f"Error: Model checkpoint does not match the model type '{cfg.model.type}'!")
        print("Please ensure the --model-type matches the architecture of the saved checkpoint.")
        return
        
    model.eval()

    print("Extracting gallery features...")
    gallery_features, gallery_v_ids, gallery_c_ids = extract_features(model, gallery_loader, device)

    out_dir = os.path.dirname(os.path.dirname(args.model_path))
    test_result_dir = os.path.join(out_dir, "test_result")
    os.makedirs(test_result_dir, exist_ok=True)

    for img_name in args.img_name:
        query_idx = -1
        for i, path in enumerate(query_img_paths):
            if os.path.basename(path) == img_name:
                query_idx = i
                break
                
        is_from_query = True
        if query_idx == -1:
            for i, path in enumerate(gallery_img_paths):
                if os.path.basename(path) == img_name:
                    query_idx = i
                    is_from_query = False
                    break
                    
        if query_idx == -1:
            print(f"Image {img_name} not found in {cfg.dataset.name} dataset.")
            continue
            
        print(f"Found {img_name} in {'query' if is_from_query else 'gallery'} dataset.")
        img_path = query_img_paths[query_idx] if is_from_query else gallery_img_paths[query_idx]
        
        dataset = query_dataset if is_from_query else gallery_dataset
        sample = dataset[query_idx]
        img_tensor = sample['img'].unsqueeze(0).to(device)
        q_v_id = sample['v_id']
        q_c_id = sample['c_id']
        
        print(f"Extracting query feature for {img_name}...")
        with torch.no_grad():
            query_feature = model(img_tensor)
            query_feature = torch.nn.functional.normalize(query_feature, p=2, dim=1).cpu()
            
        print(f"Computing similarity for {img_name}...")
        similarity = compute_similarity(query_feature, gallery_features)[0]
        
        # Mask out same-id and same-camera
        g_v_ids_np = np.array(gallery_v_ids)
        g_c_ids_np = np.array(gallery_c_ids)
        mask = (g_v_ids_np == q_v_id) & (g_c_ids_np == q_c_id)
        similarity[torch.from_numpy(mask)] = -float('inf')
        
        top_sim_score, top_index = torch.sort(similarity, descending=True)
        top_sim_score = top_sim_score[:args.top_k]
        top_index = top_index[:args.top_k]
        
        retrieval_v_ids = [gallery_v_ids[idx] for idx in top_index]
        retrieval_c_ids = [gallery_c_ids[idx] for idx in top_index]
        retrieval_img_paths = [gallery_img_paths[idx] for idx in top_index]
        
        # Visualization
        fig, ax = plt.subplots(1, args.top_k + 1, figsize=(20, 4))
        
        ax[0].imshow(plt.imread(img_path))
        ax[0].set_title(f"Query ID: {q_v_id}\nCam: {q_c_id}")
        ax[0].axis("off")
        
        for j in range(args.top_k):
            color = "green" if q_v_id == retrieval_v_ids[j] else "red"
            ax[j + 1].imshow(plt.imread(retrieval_img_paths[j]))
            ax[j + 1].set_title(f"Rank {j + 1}\nScore: {top_sim_score[j]:.3f}\nID: {retrieval_v_ids[j]}\nCam: {retrieval_c_ids[j]}", color=color)
            for spine in ax[j + 1].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
            ax[j + 1].set_xticks([])
            ax[j + 1].set_yticks([])
            
        plt.tight_layout()
        
        vis_path = os.path.join(test_result_dir, f"{img_name}_retrieval.png")
        plt.savefig(vis_path)
        plt.close()
        print(f"Saved retrieval result to {vis_path}")

if __name__ == '__main__':
    main()
