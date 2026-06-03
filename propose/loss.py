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