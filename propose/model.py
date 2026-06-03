import torch
from torch import nn
from torchvision import models
from torchvision.models import resnet50, resnet18

class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone_type = cfg.model.backbone
        self.pretrained = cfg.model.pretrained
        self.embedding_dim = cfg.model.embedding_dim
        
        dataset_cfg = cfg.dataset.vric if cfg.dataset.name == 'VRIC' else cfg.dataset.veri
        self.num_classes = dataset_cfg.num_classes

        if self.backbone_type == 'resnet50':
            from torchvision.models import resnet50
            if self.pretrained:
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            else:
                weights=None

            backbone = resnet50(weights=weights)
        
        # elif self.backbone_type == 'resnet18':
        #     from torchvision.models import resnet18
        #     if self.pretrained:
        #         weights=models.ResNet18_Weights.IMAGENET1K_V1
        #     else:
        #         weights=None

        #     backbone = resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2]) # remove avgpool and fc
        self.gap = nn.AdaptiveAvgPool2d((1,1))
                
        if self.embedding_dim != backbone.fc.out_features:
            self.embedding = nn.Linear(backbone.fc.in_features, self.embedding_dim)
            feature_dim = self.embedding_dim
        else:
            self.embedding = nn.Identity()
            feature_dim = backbone.fc.in_features

        self.classifier = nn.Linear(feature_dim, self.num_classes)

        self._init_weights()

    def summary(self, input_size=(1, 3, 256, 256)):
        device = next(self.parameters()).device
        dummy_input = torch.randn(*input_size).to(device)
        
        print(f"--- Model Summary: {self.__class__.__name__} ---")
        try:
            from torchinfo import summary
            print(summary(self, input_size=input_size, device=device))
        except ImportError:
            print("torchinfo is not installed. Please run 'pip install torchinfo' for architecture summary.")
            
        try:
            from thop import profile
            device = next(self.parameters()).device
            dummy_input = dummy_input.to(device)
            flops, params = profile(self, inputs=(dummy_input,), verbose=False)
            print(f"FLOPs: {flops / 1e9:.2f} G, Parameters: {params / 1e6:.2f} M")
        except ImportError:
            print("thop is not installed. Please run 'pip install thop' for FLOP calculation.")


    def _init_weights(self):
        if isinstance(self.embedding, nn.Linear):
            nn.init.kaiming_normal_(self.embedding.weight, mode="fan_out")
            nn.init.constant_(self.embedding.bias, 0)

            nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        # Backbone feature
        feat_map = self.backbone(x)
        
        # Global Average Pooling
        global_feat = self.gap(feat_map)
        global_feat = global_feat.view(global_feat.size(0), -1)

        if self.training:
            logits = self.classifier(global_feat)
            return {
                "metric_feat": global_feat,
                "bn_feat": global_feat, # for same output with BaselineWithBOT
                "logits": logits
            }
        else:
            return global_feat

class BaselineWithBOT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone_type = cfg.model.backbone
        self.pretrained = cfg.model.pretrained
        self.embedding_dim = cfg.model.embedding_dim
        
        dataset_cfg = cfg.dataset.vric if cfg.dataset.name == 'VRIC' else cfg.dataset.veri
        self.num_classes = dataset_cfg.num_classes

        if self.backbone_type == 'resnet50':
            from torchvision.models import resnet50
            if self.pretrained:
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            else:
                weights=None

            backbone = resnet50(weights=weights)
        
        # elif self.backbone_type == 'resnet18':
        #     from torchvision.models import resnet18
        #     if self.pretrained:
        #         weights=models.ResNet18_Weights.IMAGENET1K_V1
        #     else:
        #         weights=None

        #     backbone = resnet18(weights=weights)

        # Bag of Tricks: change stride of last layer to 1
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2]) # remove avgpool and fc
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        if self.embedding_dim != backbone.fc.out_features:
            self.embedding = nn.Linear(backbone.fc.in_features, self.embedding_dim)
            feature_dim = self.embedding_dim
        else:
            self.embedding = nn.Identity()
            feature_dim = backbone.fc.in_features

        self.bnneck = nn.BatchNorm1d(feature_dim)
        self.bnneck.bias.requires_grad = False
        
        self.classifier = nn.Linear(feature_dim, self.num_classes, bias=False)

        self._init_weights()

    def summary(self, input_size=(1, 3, 256, 256)):
        device = next(self.parameters()).device
        dummy_input = torch.randn(*input_size).to(device)
        
        print(f"--- Model Summary: {self.__class__.__name__} ---")
        try:
            from torchinfo import summary
            print(summary(self, input_size=input_size, device=device))
        except ImportError:
            print("torchinfo is not installed. Please run 'pip install torchinfo' for architecture summary.")
            
        try:
            from thop import profile
            device = next(self.parameters()).device
            dummy_input = dummy_input.to(device)
            flops, params = profile(self, inputs=(dummy_input,), verbose=False)
            print(f"FLOPs: {flops / 1e9:.2f} G, Parameters: {params / 1e6:.2f} M")
        except ImportError:
            print("thop is not installed. Please run 'pip install thop' for FLOP calculation.")


    def _init_weights(self):
        if isinstance(self.embedding, nn.Linear):
            nn.init.kaiming_normal_(self.embedding.weight, mode="fan_out")
            nn.init.constant_(self.embedding.bias, 0)

            nn.init.constant_(self.bnneck.weight, 1)
            nn.init.constant_(self.bnneck.bias, 0)

            nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        # Backbone feature
        feat_map = self.backbone(x)
        
        # Global Average Pooling
        global_feat = self.gap(feat_map)
        global_feat = global_feat.view(global_feat.size(0), -1)

        # Embedding
        feat = self.embedding(global_feat)

        # Batch Normalization
        bn_feat = self.bnneck(feat)

        if self.training:
            logits = self.classifier(bn_feat)
            return {
                "metric_feat": feat, # for metric loss
                "bn_feat": bn_feat,
                "logits": logits # for cross entropy loss
            }
        else:
            return bn_feat # for inference

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class BaselineWithCBAM(Baseline):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        if self.backbone_type == 'resnet50':
            channels = [256, 512, 1024, 2048]
        elif self.backbone_type == 'resnet18':
            channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone type for CBAM: {self.backbone_type}")

        # Append CBAM at the end of each macro layer
        self.backbone[4].add_module('cbam', CBAM(channels[0]))
        self.backbone[5].add_module('cbam', CBAM(channels[1]))
        self.backbone[6].add_module('cbam', CBAM(channels[2]))
        self.backbone[7].add_module('cbam', CBAM(channels[3]))


class BOTWithCBAM(BaselineWithBOT):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        if self.backbone_type == 'resnet50':
            channels = [256, 512, 1024, 2048]
        elif self.backbone_type == 'resnet18':
            channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone type for CBAM: {self.backbone_type}")

        # Append CBAM at the end of each macro layer
        self.backbone[4].add_module('cbam', CBAM(channels[0]))
        self.backbone[5].add_module('cbam', CBAM(channels[1]))
        self.backbone[6].add_module('cbam', CBAM(channels[2]))
        self.backbone[7].add_module('cbam', CBAM(channels[3]))