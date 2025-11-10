import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
import yaml
from typing import List, Dict, Tuple
import random
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from losses import FewShotYOLOLoss
from ultralytics.nn.modules.head import Detect 
# ==================== SUPPORT AGGREGATION MODULE ====================
class SupportAggregationModule(nn.Module):
    """
    Multi-Level Prototype Aggregation for K-shot support images
    Combines Spatial Attention + Channel Attention + Refinement
    """
    def __init__(self, channels=256):
        super().__init__()
        
        # Level 1: Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )
        
        # Level 2: Channel Attention (SE-Net inspired)
        self.channel_fc1 = nn.Conv2d(channels, channels // 16, kernel_size=1)
        self.channel_fc2 = nn.Conv2d(channels // 16, channels, kernel_size=1)
        
        # Level 3: Refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, support_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            support_features: List of [B, C, H, W], length = k_shot
        Returns:
            aggregated_support: [B, C, H, W]
        """
        k_shot = len(support_features)
        
        # Level 1: Spatial Attention Pooling
        spatial_weighted = []
        for feat in support_features:
            att_map = torch.sigmoid(self.spatial_conv(feat))  # [B, 1, H, W]
            weighted = feat * att_map
            spatial_weighted.append(weighted)
        
        F_spatial = torch.stack(spatial_weighted).mean(dim=0)
        
        # Level 2: Channel Attention Pooling
        channel_weighted = []
        for feat in support_features:
            gap = F.adaptive_avg_pool2d(feat, 1)  # [B, C, 1, 1]
            channel_att = F.relu(self.channel_fc1(gap))
            channel_att = torch.sigmoid(self.channel_fc2(channel_att))
            weighted = feat * channel_att
            channel_weighted.append(weighted)
        
        F_channel = torch.stack(channel_weighted).mean(dim=0)
        
        # Level 3: Prototype Refinement with Residual
        F_combined = self.alpha * F_spatial + self.beta * F_channel
        F_support = F_combined + self.refine_conv(F_combined)
        
        return F_support


# ==================== MATCHING MODULE ====================
class FewShotMatchingModule(nn.Module):
    """
    Few-Shot Adaptive Matching Module
    Multi-scale correlation with adaptive weighting
    Based on paper's simple matching principle
    """
    def __init__(self, channels=256):
        super().__init__()
        
        # Multi-scale correlation
        self.spatial_corr_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Correlation fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, query_feat: torch.Tensor, support_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feat: [B, C, H, W]
            support_feat: [B, C, H, W] (aggregated)
        Returns:
            matched_features: [B, C, H, W]
        """
        B, C, H, W = query_feat.shape
        
        # Corr_1: Element-wise correlation (from paper)
        corr_element = query_feat * support_feat
        
        # Corr_2: Spatial correlation
        concat_feat = torch.cat([query_feat, support_feat], dim=1)
        corr_spatial = self.spatial_corr_conv(concat_feat)
        
        # Corr_3: Global correlation
        query_global = F.adaptive_avg_pool2d(query_feat, 1)
        support_global = F.adaptive_avg_pool2d(support_feat, 1)
        corr_global = (query_global * support_global).expand(-1, -1, H, W)
        
        # Fusion
        corr_fused = torch.cat([corr_element, corr_spatial, corr_global], dim=1)
        corr_fused = self.fusion_conv(corr_fused)
        
        # Adaptive matching weights
        matching_weights = torch.sigmoid(corr_fused)
        
        # Output with residual connection (like paper: Q + ...)
        matched = query_feat + matching_weights * support_feat
        matched = self.refine(matched)
        
        return matched

class SiamYOLOMatchingModule(nn.Module):
    """
    Matching Module based on the SiamYOLOv8 paper's formula (6).
    Formula: Q_matched = Q + s(Q × S) × S
    """
    def __init__(self):
        super().__init__()
        # Không cần bất kỳ lớp nào, vì các phép toán đều có sẵn trong PyTorch
        
    def forward(self, query_feat: torch.Tensor, support_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feat: Query features [B, C, H, W]
            support_feat: Aggregated support features [B, C, H, W]
        Returns:
            matched_features: [B, C, H, W]
        """
        # 1. Channel-wise multiplication (Q × S)
        correlation_map = query_feat * support_feat
        
        # 2. Softmax to create attention map: s(Q × S)
        # Softmax được áp dụng trên các chiều không gian (H, W).
        # Ta cần reshape để áp dụng softmax rồi reshape lại.
        B, C, H, W = correlation_map.shape
        # Flatten H, W -> [B, C, H*W]
        correlation_flat = correlation_map.view(B, C, H * W)
        # Apply softmax over the spatial dimension (H*W)
        attention_map = F.softmax(correlation_flat, dim=-1)
        # Reshape back to [B, C, H, W]
        attention_map = attention_map.view(B, C, H, W)
        
        # 3. Weight the support features: s(Q × S) × S
        weighted_support = attention_map * support_feat
        
        # 4. Final fusion with residual connection: Q + ...
        matched_features = query_feat + weighted_support
        
        return matched_features

class YOLOv8Neck(nn.Module):
    """
    Neck FPN+PANet đơn giản hóa để kết hợp các feature đa tỷ lệ.
    """
    def __init__(self, in_channels_list: List[int]):
        super().__init__()
        assert len(in_channels_list) == 3
        c3_in, c4_in, c5_in = in_channels_list

        # ========== SỬA LỖI LOGIC Ở ĐÂY ==========
        # Mục tiêu là đưa tất cả các feature về cùng một số kênh để dễ xử lý.
        # Ta sẽ chọn c_out là số kênh nhỏ nhất.
        c_out = c3_in 
        
        # Các lớp Conv để giảm số kênh về c_out
        self.p5_conv1 = nn.Conv2d(c5_in, c_out, 1, 1) # p5 -> c_out
        self.p4_conv1 = nn.Conv2d(c4_in, c_out, 1, 1) # p4 -> c_out
        self.p3_conv1 = nn.Conv2d(c3_in, c_out, 1, 1) # p3 -> c_out (có thể là Identity nếu c3_in == c_out)
        
        # Các lớp Conv trong luồng Top-Down
        self.p4_conv2 = nn.Conv2d(c_out * 2, c_out, 3, 1, 1) # *2 vì concat
        self.p3_conv2 = nn.Conv2d(c_out * 2, c_out, 3, 1, 1) # *2 vì concat
        
        # Các lớp Conv trong luồng Bottom-Up
        self.n3_conv_down = nn.Conv2d(c_out, c_out, 3, 2, 1)
        self.n4_conv1 = nn.Conv2d(c_out * 2, c_out, 3, 1, 1) # *2 vì concat

        self.n4_conv_down = nn.Conv2d(c_out, c_out, 3, 2, 1)
        self.n5_conv1 = nn.Conv2d(c_out * 2, c_out, 3, 1, 1) # *2 vì concat
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features

        # ===== Luồng Top-Down (FPN) =====
        p5_td = self.p5_conv1(p5)
        p4_td = torch.cat([self.upsample(p5_td), self.p4_conv1(p4)], dim=1)
        p4_td = self.p4_conv2(p4_td)

        p3_td = torch.cat([self.upsample(p4_td), self.p3_conv1(p3)], dim=1)
        n3 = self.p3_conv2(p3_td)

        # ===== Luồng Bottom-Up (PANet) =====
        # >> CÓ THỂ LỖI NẰM Ở ĐÂY
        n4_bu = torch.cat([self.n3_conv_down(n3), p4_td], dim=1)
        n4 = self.n4_conv1(n4_bu)

        n5_bu = torch.cat([self.n4_conv_down(n4), p5_td], dim=1)
        n5 = self.n5_conv1(n5_bu)
        
        # THÊM DEBUG PRINT VÀO ĐÂY
        # print(f"[DEBUG NECK] Output shapes: n3={n3.shape}, n4={n4.shape}, n5={n5.shape}")

        return [n3, n4, n5]


# ==================== DETECTION HEAD ====================

class YOLOv8Backbone(nn.Module):
    """
    Wrapper cho backbone của YOLOv8 để trích xuất feature đa tỷ lệ.
    """
    def __init__(self, backbone_model):
        super().__init__()
        self.model = backbone_model
        # Xác định các index để lấy feature P3, P4, P5
        self.return_indices = [4, 6, 9] 
        
        # Tự động xác định số kênh đầu ra
        self.out_channels = self._get_out_channels()

    def _get_out_channels(self):
        """Tự động tính toán số kênh đầu ra từ các tầng được chọn."""
        # Chạy một tensor giả qua model để lấy shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 640, 640)
            features = []
            x = dummy_input
            for i, module in enumerate(self.model):
                x = module(x)
                if i in self.return_indices:
                    features.append(x)
        return [f.shape[1] for f in features]

    def forward(self, x):
        features = []
        for i, module in enumerate(self.model):
            x = module(x)
            if i in self.return_indices:
                features.append(x)
        return features

# ==================== MAIN MODEL ====================
class FewShotYOLO(nn.Module):
    """
    Complete Few-Shot YOLO Model
    ...
    """
    def __init__(self, 
                 backbone_weights: str = 'yolov8n.pt',
                 k_shot: int = 3,
                 freeze_backbone: bool = True,
                 reg_max: int = 16):
        super().__init__()
        self.k_shot = k_shot
        self.reg_max = reg_max
        
        # >> SỬA ĐỔI PHẦN NÀY
        # Load pretrained YOLOv8 backbone
        self.backbone = self._load_yolov8_backbone(backbone_weights)
        
        if freeze_backbone:
            self._freeze_backbone()

        # Lấy thông tin số kênh và stride một cách tự động và tường minh
        self.strides = torch.tensor([8, 16, 32])
        backbone_out_channels = self.backbone.out_channels # << LẤY SỐ KÊNH TỰ ĐỘNG
        print(f"✓ Detected backbone output channels: {backbone_out_channels}")
        
        # 1. Tạo các Module Aggregation riêng cho mỗi scale
        # Sử dụng nn.ModuleList để PyTorch có thể nhận diện các module này
        self.support_aggregation_modules = nn.ModuleList()
        for channels in backbone_out_channels:
            self.support_aggregation_modules.append(SupportAggregationModule(channels=channels))
        print(f"✓ Created {len(self.support_aggregation_modules)} Support Aggregation Modules")

        # 2. Tạo các Module Matching riêng cho mỗi scale
        # (Sử dụng SiamYOLOMatchingModule tôi đã gợi ý ở lần trước)
        self.matching_modules = nn.ModuleList()
        for _ in backbone_out_channels: # Matching module không cần tham số channels
             self.matching_modules.append(SiamYOLOMatchingModule())
        print(f"✓ Created {len(self.matching_modules)} SiamYOLO Matching Modules")

        # 3. Khởi tạo Neck và Head (giữ nguyên hoặc sửa đổi nếu cần)
        self.neck = YOLOv8Neck(in_channels_list=backbone_out_channels)
        
        # Lấy số kênh đầu ra của Neck để khởi tạo Head
        # Giả sử Neck trả về các feature có cùng số kênh là `neck_out_channel`
        neck_out_channel = backbone_out_channels[0] 
        head_in_channels = [neck_out_channel] * 3
        self.detection_head = Detect(nc=1, ch=head_in_channels)
        
        self.detection_head.stride = self.strides.to(next(self.detection_head.parameters()).device)
        self.detection_head.bias_init() # Gọi hàm khởi tạo bias
        print("✓ Đã gọi bias_init() cho Detect Head.")

        print(f"✓ Model initialized with {k_shot}-shot support")
        print(f"✓ Backbone frozen: {freeze_backbone}")
        print(f"✓ Multi-scale architecture enabled.")
        print(f"✓ Trainable params: {self._count_trainable_params():,}")
        # << KẾT THÚC SỬA ĐỔI

    def _load_yolov8_backbone(self, weights_path: str):
        """Load YOLOv8 backbone from Ultralytics"""
        try:
            from ultralytics import YOLO
            model = YOLO(weights_path)
            # >> SỬA ĐỔI ĐOẠN NÀY
            # Trích xuất backbone (thường là 10 layer đầu)
            backbone_layers = model.model.model[:10]
            # Bọc backbone trong wrapper để lấy intermediate features
            return YOLOv8Backbone(backbone_layers)
            # << KẾT THÚC SỬA ĐỔI
        except Exception as e:
            print(f"Warning: Using simple backbone (Ultralytics not available or error: {e})")
            # Cần sửa cả backbone đơn giản để trả về 3 feature
            raise NotImplementedError("Simple backbone needs to be updated for multi-scale output.")
    
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen")
    def _count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, support_imgs: torch.Tensor, query_img: torch.Tensor):
        B, K, C, H, W = support_imgs.shape

        # 1. Trích xuất đặc trưng backbone
        query_features_list = self.backbone(query_img)
        support_features_per_scale = [[] for _ in range(len(query_features_list))]
        for k in range(K):
            with torch.no_grad():
                support_feat_list = self.backbone(support_imgs[:, k])
            for i, feat in enumerate(support_feat_list):
                support_features_per_scale[i].append(feat)

        # 2. Aggregate & Matching (TRƯỚC NECK)
        matched_features_list = []
        num_scales = len(query_features_list)
        
        for i in range(num_scales):
            # Lấy module thứ i từ ModuleList
            aggregation_module = self.support_aggregation_modules[i]
            
            # <<< DÒNG NÀY SẼ GỌI ĐÚNG MODULE SIAMYOLO MÀ BẠN VỪA TẠO >>>
            matching_module = self.matching_modules[i] 

            # Gộp support features
            current_scale_supports = support_features_per_scale[i]
            aggregated_support = aggregation_module(current_scale_supports) 
            
            # Matching
            query_feat_scale_i = query_features_list[i]
            matched_feat = matching_module(query_feat_scale_i, aggregated_support)
            matched_features_list.append(matched_feat)
        
        # 3. Đưa vào Neck & Head
        neck_outputs = self.neck(matched_features_list)
        # print(f"[DEBUG FORWARD] Neck output shapes: {[f.shape for f in neck_outputs]}")
        head_outputs = self.detection_head(neck_outputs)
        if self.training:
        # Khi training, head trả về list feature map thô
            return head_outputs
        else:
            # Khi eval, head trả về tuple (processed_preds, raw_feature_maps)
            # Chúng ta chỉ lấy phần thứ hai (raw_feature_maps) để đưa vào loss
            return head_outputs
        # =======================================================
        
        return predictions_list


# ==================== LOSS FUNCTIONS ====================


# ==================== DATASET ====================
class FewShotDataset(Dataset):
    """
    Few-Shot Detection Dataset
    Episode-based sampling for N-way K-shot training
    """
    def __init__(self, 
                 data_root: str,
                 n_way: int = 5,
                 k_shot: int = 3,
                 n_query: int = 5,
                 img_size: int = 640,
                 mode: str = 'train'):
        
        self.data_root = Path(data_root)
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.img_size = img_size
        self.mode = mode
        
        # Load class directories
        self.classes = self._load_classes()
        
        # Augmentation
        self.transform = self._get_transforms()
        
        print(f"✓ Dataset loaded: {len(self.classes)} classes")
    
    def _load_classes(self) -> List[Path]:
        """Load all class directories"""
        class_dirs = []
        mode_path = self.data_root / self.mode
        
        for class_dir in mode_path.iterdir():
            if class_dir.is_dir():
                support_dir = class_dir / 'support'
                query_dir = class_dir / 'query'
                
                if support_dir.exists() and query_dir.exists():
                    class_dirs.append(class_dir)
        
        return class_dirs
    
    def _get_transforms(self):
        """Data augmentation"""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def _load_image_and_label(self, img_path: Path):
        """Load image and YOLO format label"""
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load label
        label_path = img_path.with_suffix('.txt')
        boxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        boxes.append([x, y, w, h])
                        class_labels.append(int(class_id))
        
        return img, boxes, class_labels
    
    def __len__(self):
        return 1000  # Number of episodes per epoch
    
    def __getitem__(self, idx):
        """Sample one episode: N-way K-shot"""
        # Sample N random classes
        episode_classes = random.sample(self.classes, self.n_way)
        
        support_images = []
        query_images = []
        query_targets = []
        
        for class_idx, class_dir in enumerate(episode_classes):
            # Get support and query image paths
            support_dir = class_dir / 'support'
            query_dir = class_dir / 'query'
            
            support_imgs = list(support_dir.glob('*.jpg')) + list(support_dir.glob('*.png'))
            query_imgs = list(query_dir.glob('*.jpg')) + list(query_dir.glob('*.png'))
            
            # Sample K support images
            sampled_support = random.sample(support_imgs, min(self.k_shot, len(support_imgs)))
            
            for img_path in sampled_support:
                img, boxes, labels = self._load_image_and_label(img_path)
                
                # Apply transform
                transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
                support_images.append(transformed['image'])
            
            # Sample query images
            sampled_query = random.sample(query_imgs, min(self.n_query, len(query_imgs)))
            
            for img_path in sampled_query:
                img, boxes, labels = self._load_image_and_label(img_path)
                
                transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
                query_images.append(transformed['image'])
                
                # Create target dict
                target = self._create_target(transformed['bboxes'], transformed['class_labels'])
                query_targets.append(target)
        
        # Stack tensors
        support_batch = torch.stack(support_images[:self.k_shot])  # [K, 3, H, W]
        
        # Return first query for simplicity
        if len(query_images) > 0:
            query_image = query_images[0]
            query_target = query_targets[0]
        else:
            # Fallback
            query_image = support_batch[0]
            query_target = self._create_empty_target()
        
        return {
            'support_images': support_batch,
            'query_image': query_image,
            'query_target': query_target
        }
    
    def _create_target(self, boxes, labels):
        """Create target dict from boxes and labels"""
        num_boxes = len(boxes)
        # Simplified - expand as needed
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0),
            'obj': torch.ones(num_boxes, dtype=torch.float32),  # Sửa ở đây: shape [num_boxes]
            'cls': torch.ones(num_boxes, dtype=torch.float32)   # Sửa ở đây: shape [num_boxes]
    }
    
    def _create_empty_target(self):
        """Create empty target"""
        return {
            'boxes': torch.zeros(0, 4),
            'labels': torch.zeros(0),
            'obj': torch.zeros(1),
            'cls': torch.zeros(1)
        }


# ==================== TRAINING ====================

def fewshot_collate_fn(batch):
    # batch là một list các dictionary trả về từ __getitem__
    support_images = torch.stack([item['support_images'] for item in batch], dim=0)
    query_images = torch.stack([item['query_image'] for item in batch], dim=0)
    
    # Giữ query_targets là một list các dictionary
    query_targets = [item['query_target'] for item in batch]
    
    return {
        'support_images': support_images,
        'query_image': query_images,
        'query_target': query_targets
    }

class FewShotTrainer:
    """
    Trainer for Few-Shot YOLO
    Handles training loop, validation, and checkpointing
    """
    def __init__(self,
                 model: FewShotYOLO,
                 train_dataset: FewShotDataset,
                 val_dataset: FewShotDataset,
                 lr: float = 1e-4,
                 batch_size: int = 4,
                 epochs: int = 100,
                 device: str = 'cuda',
                 save_dir: str = 'checkpoints'):
        
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.epochs = epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=fewshot_collate_fn 
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=fewshot_collate_fn 
        )
        
        # Loss and optimizer
        self.criterion = FewShotYOLOLoss(model)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=0.0001
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        print("✓ Trainer initialized")
    
    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            support_imgs = batch['support_images'].to(self.device)
            query_img = batch['query_image'].to(self.device)
            query_target_list = [{k: v.to(self.device) for k, v in t.items()} for t in batch['query_target']]
            
            # Forward
            self.optimizer.zero_grad()
            predictions_list = self.model(support_imgs, query_img)
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions_list, query_target_list)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ciou': f"{loss_dict.get('ciou', 0):.3f}",
                'dfl': f"{loss_dict.get('dfl', 0):.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                support_imgs = batch['support_images'].to(self.device)
                query_img = batch['query_image'].to(self.device)
                query_target_list = [{k: v.to(self.device) for k, v in t.items()} for t in batch['query_target']]
                
                _, predictions_list = self.model(support_imgs, query_img)
                loss, _ = self.criterion(predictions_list, query_target_list)   
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved at epoch {epoch}")
    
    def train(self):
        """Complete training loop"""
        print("=" * 60)
        print("Starting Few-Shot YOLO Training")
        print("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            # Print summary
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"LR: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            print("-" * 60)
        
        print("\n✓ Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(self.epochs, is_best=False)


# ==================== INFERENCE ====================
class FewShotInference:
    """
    Inference pipeline for Few-Shot YOLO
    Handles detection with K-shot support images
    """
    def __init__(self,
                 model: FewShotYOLO,
                 checkpoint_path: str,
                 device: str = 'cuda',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        
        self.model = model.to(device)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.reg_max = self.model.reg_max
        self.proj = torch.arange(self.reg_max + 1, device=self.device, dtype=torch.float)
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Preprocessing
        self.transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print("✓ Inference pipeline ready")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
    
    def preprocess_image(self, img_path: str) -> torch.Tensor:
        """Preprocess single image"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image=img)
        img_tensor = transformed['image'].unsqueeze(0)  # [1, 3, H, W]
        
        return img_tensor, img
    
    def detect(self,
               support_image_paths: List[str],
               query_image_path: str) -> Dict:
        """
        Perform detection on query image using support images
        
        Args:
            support_image_paths: List of K support image paths
            query_image_path: Query image path to detect objects
            
        Returns:
            results: Dict with boxes, scores, and visualization
        """
        self.model.eval()
        
        # Load and preprocess support images
        support_tensors = []
        for img_path in support_image_paths:
            img_tensor, _ = self.preprocess_image(img_path)
            support_tensors.append(img_tensor)
        
        support_batch = torch.cat(support_tensors, dim=0)  # [K, 3, H, W]
        support_batch = support_batch.unsqueeze(0).to(self.device)  # [1, K, 3, H, W]
        
        # Load and preprocess query image
        query_tensor, query_img_orig = self.preprocess_image(query_image_path)
        query_tensor = query_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            self.model.eval() 
            predictions_decoded = self.model.detection_head(matched_features_list)
        
        # Post-process predictions
        boxes, scores, class_ids = self._postprocess(predictions_decoded, query_img_orig.shape[:2])
        
        # Visualize
        vis_img = self._visualize(query_img_orig, boxes, scores)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'visualization': vis_img
        }
    
    def _postprocess(self, predictions: torch.Tensor, orig_shape: Tuple[int, int]):
        """
        Post-process predictions từ Detect head của Ultralytics.
        """
        # predictions shape: [batch_size, num_proposals, 4 (xywh) + 1 (score)]
        # Vì batch_size là 1 khi inference, ta lấy predictions[0]
        preds = predictions[0] # [num_proposals, 5]
        
        # Lọc theo confidence
        preds = preds[preds[:, 4] > self.conf_threshold]
        if len(preds) == 0:
            return np.array([]), np.array([]), np.array([])
            
        boxes = preds[:, :4] # xywh
        scores = preds[:, 4]
        
        # Chuyển box về xyxy và scale về kích thước ảnh gốc
        boxes = self._xywh_to_xyxy(boxes)
        boxes = self._scale_boxes(boxes, (640, 640), orig_shape)
        
        # NMS
        keep_indices = self._nms(boxes, scores, self.iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = torch.zeros(len(boxes), dtype=torch.long)
        
        return boxes.cpu().numpy(), scores.cpu().numpy(), class_ids.cpu().numpy()
    
    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from xywh to xyxy format"""
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return boxes_xyxy
    
    def _scale_boxes(self, boxes: torch.Tensor, from_shape: Tuple, to_shape: Tuple) -> torch.Tensor:
        """Scale boxes from one shape to another"""
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
        pad_x = (from_shape[1] - to_shape[1] * gain) / 2
        pad_y = (from_shape[0] - to_shape[0] * gain) / 2
        
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain
        
        # Clip to image boundaries
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, to_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, to_shape[0])
        
        return boxes
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Non-Maximum Suppression"""
        from torchvision.ops import nms
        keep = nms(boxes, scores, iou_threshold)
        return keep
    
    def _visualize(self, img: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Visualize detections on image"""
        vis_img = img.copy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Object: {score:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_img
    
    def save_results(self, results: Dict, output_path: str):
        """Save detection results"""
        vis_img = results['visualization']
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_img_bgr)
        print(f"✓ Results saved to {output_path}")


# ==================== MAIN SCRIPT ====================
def main():
    """Main execution script"""
    
    # ============= CONFIGURATION =============
    CONFIG = {
        # Data
        'data_root': '/mlcv2/WorkingSpace/Personal/chinhnm/NL_fewshot/data/',
        'n_way': 7,
        'k_shot': 3,
        'n_query': 5,
        'img_size': 640,
        
        # Model
        'backbone_weights': 'yolov8n.pt',
        'freeze_backbone': True,
        
        # Training
        'batch_size': 4,
        'lr': 1e-4,
        'epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Paths
        'save_dir': '/mlcv2/WorkingSpace/Personal/chinhnm/NL_fewshot/checkpoints/',
        'results_dir': '/mlcv2/WorkingSpace/Personal/chinhnm/NL_fewshot/results/'
    }
    
    print("=" * 60)
    print("Few-Shot YOLO - Complete Pipeline")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"K-shot: {CONFIG['k_shot']}")
    print(f"Backbone frozen: {CONFIG['freeze_backbone']}")
    print("=" * 60)
    
    # ============= TRAINING =============
    print("\n[1] Initializing datasets...")
    train_dataset = FewShotDataset(
        data_root=CONFIG['data_root'],
        n_way=CONFIG['n_way'],
        k_shot=CONFIG['k_shot'],
        n_query=CONFIG['n_query'],
        img_size=CONFIG['img_size'],
        mode='train'
    )
    
    val_dataset = FewShotDataset(
        data_root=CONFIG['data_root'],
        n_way=CONFIG['n_way'],
        k_shot=CONFIG['k_shot'],
        n_query=CONFIG['n_query'],
        img_size=CONFIG['img_size'],
        mode='val'
    )
    
    print("\n[2] Initializing model...")
    model = FewShotYOLO(
        backbone_weights=CONFIG['backbone_weights'],
        k_shot=CONFIG['k_shot'],
        freeze_backbone=CONFIG['freeze_backbone']
    )
    
    print("\n[3] Starting training...")
    trainer = FewShotTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=CONFIG['lr'],
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        device=CONFIG['device'],
        save_dir=CONFIG['save_dir']
    )
    
    trainer.train()
    
    print("\n✓ Training completed successfully!")


# ==================== INFERENCE SCRIPT ====================
def inference_example():
    """Example inference script"""
    
    print("=" * 60)
    print("Few-Shot YOLO - Inference Example")
    print("=" * 60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'checkpoints/best_model.pth'
    
    # Initialize model
    print("\n[1] Loading model...")
    model = FewShotYOLO(
        backbone_weights='yolov8n.pt',
        k_shot=3,
        freeze_backbone=True
    )
    
    # Initialize inference pipeline
    inference = FewShotInference(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Prepare support and query images
    support_images = [
        'examples/support_1.jpg',
        'examples/support_2.jpg',
        'examples/support_3.jpg'
    ]
    query_image = 'examples/query.jpg'
    
    print("\n[2] Running detection...")
    results = inference.detect(support_images, query_image)
    
    # Print results
    print(f"\n[3] Results:")
    print(f"Detected {len(results['boxes'])} objects")
    for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
        print(f"  Object {i+1}: Box={box}, Score={score:.3f}")
    
    # Save results
    output_path = 'results/detection_result.jpg'
    Path('results').mkdir(exist_ok=True)
    inference.save_results(results, output_path)
    
    print("\n✓ Inference completed!")


if __name__ == '__main__':
    # Uncomment the mode you want to run
    
    # Mode 1: Training
    main()
    
    # Mode 2: Inference
    # inference_example()