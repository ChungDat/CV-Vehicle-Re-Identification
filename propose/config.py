import os
import time
from ml_collections import ConfigDict
import torch

def get_config():
    cfg = ConfigDict()

    # ======
    # System
    # ======
    cfg.system = ConfigDict()

    cfg.system.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.system.num_workers = 4 # for dataloader
    cfg.system.pin_memory = True # for dataloader
    
    # ======
    # Model
    # ======
    cfg.model = ConfigDict()

    cfg.model.type = 'baseline' # [baseline, baseline_with_BOT, baseline_with_CBAM, BOT_with_CBAM]

    cfg.model.backbone = 'resnet50'
    cfg.model.pretrained = True
    cfg.model.embedding_dim = 2048

    # ======
    # Experiment
    # ======
    cfg.seed = 42
    cfg.output_dir = './outputs'
    
    timestamp = time.strftime('%Y_%m_%d-%H_%M_%S')
    cfg.save_dir = os.path.join(cfg.output_dir, cfg.model.type, timestamp)

    # ======
    # Dataset
    # ======
    cfg.dataset = ConfigDict()

    cfg.dataset.name = "VeRi776" # [VeRi776, VRIC]

    # VeRi776 dataset
    cfg.dataset.veri = ConfigDict()
    cfg.dataset.veri.root = "./datasets/VeRi776"
    cfg.dataset.veri.train_dir = os.path.join(cfg.dataset.veri.root, "image_train")
    cfg.dataset.veri.query_dir = os.path.join(cfg.dataset.veri.root, "image_query")
    cfg.dataset.veri.gallery_dir = os.path.join(cfg.dataset.veri.root, "image_test")
    cfg.dataset.veri.num_classes = 776

    # VRIC dataset
    cfg.dataset.vric = ConfigDict()
    cfg.dataset.vric.root = "./datasets/VRIC"
    cfg.dataset.vric.train_dir = os.path.join(cfg.dataset.vric.root, "train_images")
    cfg.dataset.vric.query_dir = os.path.join(cfg.dataset.vric.root, "probe_images")
    cfg.dataset.vric.gallery_dir = os.path.join(cfg.dataset.vric.root, "gallery_images")
    cfg.dataset.vric.train_list = os.path.join(cfg.dataset.vric.root, "vric_train.txt")
    cfg.dataset.vric.query_list = os.path.join(cfg.dataset.vric.root, "vric_probe.txt")
    cfg.dataset.vric.gallery_list = os.path.join(cfg.dataset.vric.root, "vric_gallery.txt")
    cfg.dataset.vric.num_classes = 2811

    cfg.dataset.image_size = (256, 256)

    # ======
    # Data Augmentation
    # ======
    cfg.augmentation = ConfigDict()

    cfg.augmentation.prob = 0.5

    # General augmentations
    cfg.augmentation.random_flip = True
    cfg.augmentation.color_jitter = True
    cfg.augmentation.random_affine = True
    
    # ======
    # Loss
    # ======
    cfg.loss = ConfigDict()
    cfg.loss.metric_loss = 'triplet' # [triplet, circle, center]
    cfg.loss.metric_margin = 0.3

    cfg.loss.cls_weight = 1.0
    cfg.loss.metric_weight = 1.0

    # ======
    # Optimizer
    # ======
    cfg.optimizer = ConfigDict()

    cfg.optimizer.type = 'adamw'
    
    cfg.optimizer.base_lr = 0.0003
    cfg.optimizer.backbone_lr_mult = 0.1 # Backbone LR is 10x smaller than base_lr
    cfg.optimizer.weight_decay = 5e-4

    # ====== 
    # Scheduler
    # ======
    cfg.scheduler = ConfigDict()
    
    cfg.scheduler.type = 'cosine'
    
    cfg.scheduler.min_lr = 1e-5

    # ======
    # Training
    # ======
    cfg.training = ConfigDict()
    
    cfg.training.total_epochs = 120
    cfg.training.batch_size = 32
    cfg.training.num_instances = 4 # K instances per ID for Triplet Loss, e.g. batch size = 32, num_instances = 4 means 8 IDs per batch

    cfg.training.save_every = 10 # Save checkpoint every 10 epochs
    cfg.training.eval_every = 10 # Evaluate every 10 epochs
    cfg.training.print_every = 10 # Print training info every 10 steps (not epochs)

    cfg.training.patience = 10 # for early stopping
    cfg.training.min_delta = 0.001 # for early stopping
    
    # ====== 
    # Evaluation & Checkpoint
    # ====== 
    cfg.eval = ConfigDict()
    cfg.eval.metrics = ['mAP', 'Rank-1', 'Rank-5', 'Rank-10'] # [mAP, Rank-1, Rank-5, Rank-10]

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.save_dir = os.path.join(cfg.save_dir, "checkpoint")
    cfg.checkpoint.best_model = os.path.join(cfg.checkpoint.save_dir, "best_model.pth")

    # ======
    # Plot
    # ======
    cfg.plot = ConfigDict()
    cfg.plot.save_fig = True
    cfg.plot.save_dir = os.path.join(cfg.save_dir, "plots")
    cfg.plot.fig_name = "metrics.png"

    # ======
    # Logging
    # ======
    cfg.logging = ConfigDict()
    cfg.logging.save_dir = os.path.join(cfg.save_dir, "logs")

    # ======
    # Bag of Tricks
    # ======
    # Warm up learning rate
    cfg.scheduler.warmup_epochs = 10 # 0 to disable
    cfg.scheduler.warmup_lr = 1e-6
    
    # Random erasing
    cfg.augmentation.random_erasing = False
    
    # Label smoothing
    cfg.loss.label_smoothing = False
    
    return cfg