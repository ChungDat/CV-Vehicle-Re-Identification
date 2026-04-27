from load_dataset import RandomTransform
from load_dataset import ValTransform
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch.backends import cudnn

from config import get_config
from load_dataset import VeRiDataset, VRICDataset, RandomTransform, ValTransform
from sampler import RandomIdentitySampler
from model import Baseline, BaselineWithBOT, BaselineWithCBAM, BOTWithCBAM
from train_one_epoch import train_one_epoch
from utils import *

class ReIDLoss(nn.Module):
    def __init__(self, cfg, label_encoder, device):
        super().__init__()
        self.triplet_loss = TripletLoss(cfg)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1 if getattr(cfg.loss, 'label_smoothing', False) else 0.0)
        self.cls_weight = getattr(cfg.loss, 'cls_weight', 1.0)
        self.metric_weight = getattr(cfg.loss, 'metric_weight', 1.0)
        self.label_encoder = label_encoder
        self.device = device

    def forward(self, preds, labels):
        # convert string labels to ints
        if isinstance(labels, torch.Tensor):
            int_labels = labels
        else:
            int_labels = torch.tensor([self.label_encoder[vid] for vid in labels], dtype=torch.long, device=self.device)
            
        if isinstance(preds, dict):
            logits = preds['logits']
            c_loss = self.ce_loss(logits, int_labels)
            
            if 'metric_feat' in preds:
                metric_feat = preds['metric_feat']
                t_loss = self.triplet_loss(metric_feat, int_labels)
                total_loss = self.metric_weight * t_loss + self.cls_weight * c_loss
                return total_loss, t_loss, c_loss
            else:
                total_loss = self.cls_weight * c_loss
                return total_loss, c_loss
        else:
            # If preds is just a tensor (logits)
            logits = preds
            c_loss = self.ce_loss(logits, int_labels)
            total_loss = self.cls_weight * c_loss
            return total_loss, c_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train ReID model")
    parser.add_argument('--dataset-name', type=str, default=None, help='Dataset name [VeRi776, VRIC]')
    parser.add_argument('--model-type', type=str, default=None, help='Model type [baseline, baseline_with_BOT, baseline_with_CBAM, BOT_with_CBAM]')
    parser.add_argument('--metric-loss', type=str, default=None, help='Metric loss [triplet, circle]')
    parser.add_argument('--total-epochs', type=int, default=None, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--num-instances', type=int, default=None, help='Number of instances per identity in a batch')
    parser.add_argument('--warmup-epochs', type=int, default=None, help='Warm up learning rate epochs (0 to disable)')
    parser.add_argument('--random-erasing', type=str, default=None, choices=['True', 'False', 'true', 'false'], help='Turn on random erasing (True/False)')
    parser.add_argument('--label-smoothing', type=str, default=None, choices=['True', 'False', 'true', 'false'], help='Turn on label smoothing (True/False)')
    return parser.parse_args()

def main():
    cfg = get_config()
    args = parse_args()
    
    if args.dataset_name is not None:
        cfg.dataset.name = args.dataset_name

    dataset_cfg = cfg.dataset.vric if cfg.dataset.name == 'VRIC' else cfg.dataset.veri

    if args.model_type is not None:
        cfg.model.type = args.model_type

        timestamp = os.path.basename(cfg.save_dir)
        cfg.save_dir = os.path.join(cfg.output_dir, cfg.model.type, timestamp)
        cfg.checkpoint.save_dir = os.path.join(cfg.save_dir, "checkpoint")
        cfg.checkpoint.best_model = os.path.join(cfg.checkpoint.save_dir, "best_model.pth")
        cfg.plot.save_dir = os.path.join(cfg.save_dir, "plots")
        cfg.logging.save_dir = os.path.join(cfg.save_dir, "logs")
    if args.metric_loss is not None:
        cfg.loss.metric_loss = args.metric_loss
    if args.total_epochs is not None:
        cfg.training.total_epochs = args.total_epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.num_instances is not None:
        cfg.training.num_instances = args.num_instances
        
    # Bag of Tricks
    if args.warmup_epochs is not None:
        cfg.scheduler.warmup_epochs = args.warmup_epochs
    if args.random_erasing is not None:
        cfg.augmentation.random_erasing = (args.random_erasing.lower() == 'true')
    if args.label_smoothing is not None:
        cfg.loss.label_smoothing = (args.label_smoothing.lower() == 'true')
    
    # Fix random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    # Directories
    if not os.path.isdir(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    if not os.path.isdir(cfg.logging.save_dir):
        os.makedirs(cfg.logging.save_dir)
    if not os.path.isdir(cfg.checkpoint.save_dir):
        os.makedirs(cfg.checkpoint.save_dir)
        
    logger = logger_config(os.path.join(cfg.logging.save_dir, 'train.log'))
    logger.info("========== Starting Training ==========")
    logger.info(f"Config: {cfg}")
    
    device = torch.device(cfg.system.device)
    
    # Data loaders
    train_transform = RandomTransform(cfg)
    val_transform = ValTransform(cfg)
    
    if cfg.dataset.name == 'VRIC':
        train_dataset = VRICDataset(dataset_cfg.train_dir, dataset_cfg.train_list, transform=train_transform)
        query_dataset = VRICDataset(dataset_cfg.query_dir, dataset_cfg.query_list, transform=val_transform)
        gallery_dataset = VRICDataset(dataset_cfg.gallery_dir, dataset_cfg.gallery_list, transform=val_transform)
    else:
        train_dataset = VeRiDataset(dataset_cfg.train_dir, transform=train_transform)
        query_dataset = VeRiDataset(dataset_cfg.query_dir, transform=val_transform)
        gallery_dataset = VeRiDataset(dataset_cfg.gallery_dir, transform=val_transform)
    
    # Label encoder
    unique_ids = sorted(list(set([v_id for _, v_id, _, _ in train_dataset.samples])))
    label_encoder = {v_id: idx for idx, v_id in enumerate(unique_ids)}
    
    sampler = RandomIdentitySampler(train_dataset, batch_size=cfg.training.batch_size, num_instances=cfg.training.num_instances)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
        drop_last=True
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory
    )
    
    # Model
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
    
    # Loss
    criterion = ReIDLoss(cfg, label_encoder, device)
    
    # Optimizer
    base_lr = cfg.optimizer.base_lr
    backbone_lr_mult = getattr(cfg.optimizer, 'backbone_lr_mult', 0.1)
    
    backbone_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            new_params.append(param)
            
    param_groups = [
        {'params': backbone_params, 'lr': base_lr * backbone_lr_mult},
        {'params': new_params, 'lr': base_lr}
    ]

    if cfg.optimizer.type == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.optimizer.weight_decay)
    else:
        optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.optimizer.weight_decay)
        
    # Scheduler
    if cfg.scheduler.type == 'cosine':
        if getattr(cfg.scheduler, 'warmup_epochs', 0) > 0:
            warmup_epochs = cfg.scheduler.warmup_epochs
            warmup_factor = cfg.scheduler.warmup_lr / cfg.optimizer.base_lr
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, end_factor=1.0, total_iters=warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.total_epochs - warmup_epochs, eta_min=cfg.scheduler.min_lr
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.total_epochs, eta_min=cfg.scheduler.min_lr
            )
    else:
        lr_scheduler = None
        
    writer = SummaryWriter(cfg.logging.save_dir)
    early_stopper = EarlyStopper(cfg)
    
    best_mAP = 0.0
    
    history = {
        'train_loss': [],
        'epochs': [],
        'eval_epochs': []
    }
    
    for metric in cfg.eval.metrics:
        history[metric] = []
    
    for epoch in range(1, cfg.training.total_epochs + 1):
        logger.info(f'\n========= Epoch [{epoch}/{cfg.training.total_epochs}] =========')
        
        train_loss = train_one_epoch(
            data_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            logger=logger,
            device=device,
            cfg=cfg
        )
        
        history['train_loss'].append(train_loss)
        history['epochs'].append(epoch)
        
        is_best = False
        should_stop = False
        
        if epoch % cfg.training.eval_every == 0 or epoch == cfg.training.total_epochs:
            logger.info("Evaluating...")
            cmc, mAP = evaluate(model, query_loader, gallery_loader, device)
            
            logger.info(f"Evaluation Metrics:")
            
            for metric in cfg.eval.metrics:
                if metric == 'mAP':
                    val = mAP
                elif metric.startswith('Rank-'):
                    rank = int(metric.split('-')[1])
                    val = cmc[rank - 1]
                else:
                    continue
                
                logger.info(f"{metric}: {val:.4f}")
                writer.add_scalar(f'Eval/{metric}', val, epoch)
                history[metric].append(val)
                
            history['eval_epochs'].append(epoch)
                
            if mAP > best_mAP:
                best_mAP = mAP
                logger.info(f"New best mAP: {best_mAP:.4f}")
                is_best = True
                
            # Negate mAP because EarlyStopper expects validation "loss" to decrease
            if early_stopper.early_stop(-mAP): 
                logger.info(f"Early stopping triggered at epoch {epoch}")
                should_stop = True
                
        # Save checkpoint if it's the best model OR if it matches save_every frequency
        if is_best or epoch % cfg.training.save_every == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mAP': best_mAP,
            }, cfg.checkpoint.save_dir, is_best=is_best)
            
        save_history(history, cfg)
            
        if should_stop:
            break

if __name__ == '__main__':
    main()
