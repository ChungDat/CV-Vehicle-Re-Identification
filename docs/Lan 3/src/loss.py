import torch
from torch import nn
from utils import compute_similarity

class TripletLoss(nn.Module):
    def __init__(self, cfg, distance_mode='cosine'):
        super().__init__()
        self.margin = cfg.loss.metric_margin
        self.distance_mode = distance_mode

    def forward(self, features, labels):
        n = features.size(0)

        if self.distance_mode == 'cosine':
            # L2 normalize
            features = nn.functional.normalize(features, p=2, dim=1)
            similarity_matrix = compute_similarity(features, features)
            # Cosine Distance
            dist_matrix = 1 - similarity_matrix
        elif self.distance_mode == 'euclid':
            dist_matrix = torch.cdist(features, features, p=2)
        else:
            raise ValueError(f"Unsupported distance mode: {self.distance_mode}")

        labels = labels.unsqueeze(1)

        mask_pos = (labels == labels.t())
        mask_pos.fill_diagonal_(False)
        mask_neg = (labels != labels.t())

        # Hardest positive
        dist_ap = dist_matrix.masked_fill(~mask_pos, -1.0).max(dim=1)[0]

        # Hardest negative
        dist_an = dist_matrix.masked_fill(~mask_neg, float('inf')).min(dim=1)[0]

        # loss = max(0, dp - dn + m)
        loss = torch.relu(
            dist_ap - dist_an + self.margin
        )

        return loss.mean()

class CircleLoss(nn.Module):
    def __init__(self, cfg, distance_mode='cosine'):
        raise NotImplementedError("CircleLoss is not implemented yet.")

class CenterLoss(nn.Module):
    def __init__(self, cfg, distance_mode='cosine'):
        raise NotImplementedError("CenterLoss is not implemented yet.")

class ReIDLoss(nn.Module):
    def __init__(self, cfg, label_encoder, device):
        super().__init__()
        metric_loss_type = getattr(cfg.loss, 'metric_loss', 'triplet')
        distance_mode = getattr(cfg.loss, 'distance_mode', 'cosine')
        
        if metric_loss_type == 'triplet':
            self.metric_loss = TripletLoss(cfg, distance_mode=distance_mode)
        elif metric_loss_type == 'circle':
            self.metric_loss = CircleLoss(cfg, distance_mode=distance_mode)
        elif metric_loss_type == 'center':
            self.metric_loss = CenterLoss(cfg, distance_mode=distance_mode)
        else:
            print(f"Warning: Metric loss {metric_loss_type} not implemented yet, using triplet loss")
            self.metric_loss = TripletLoss(cfg, distance_mode=distance_mode)
            
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
                m_loss = self.metric_loss(metric_feat, int_labels)
                total_loss = self.metric_weight * m_loss + self.cls_weight * c_loss
                return total_loss, m_loss, c_loss
            else:
                total_loss = self.cls_weight * c_loss
                return total_loss, c_loss
        else:
            # If preds is just a tensor (logits)
            logits = preds
            c_loss = self.ce_loss(logits, int_labels)
            total_loss = self.cls_weight * c_loss
            return total_loss, c_loss