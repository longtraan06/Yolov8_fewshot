"""
Few-Shot YOLO Detection Model
Based on SiamYOLOv8 paper with 3-shot support
Author: AI Assistant
"""

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

# ==================== SUPPORT AGGREGATION MODULE ====================
class SupportAggregationModule(nn.Module):
    """
    Multi-Level Prototype Aggregation for K-shot support images
    Combines Spatial Attention + Channel Attention + Refinement
    """
    def __init__(self, channels=512):
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
    def __init__(self, channels=512):
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


# ==================== DETECTION HEAD ====================
class FewShotDetectionHead(nn.Module):
    """
    Detection head for few-shot object detection
    Single-class detection following paper's approach
    """
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        
        # Detection branches
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Outputs
        self.box_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=1)  # bbox
        self.obj_pred = nn.Conv2d(256, num_anchors * 1, kernel_size=1)  # objectness
        self.cls_pred = nn.Conv2d(256, num_anchors * 1, kernel_size=1)  # class (single-class)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            predictions: Dict with 'boxes', 'obj', 'cls'
        """
        feat = self.conv1(x)
        feat = self.conv2(feat)
        
        B, _, H, W = feat.shape
        
        # Predictions
        box_pred = self.box_pred(feat)  # [B, num_anchors*4, H, W]
        obj_pred = self.obj_pred(feat)  # [B, num_anchors*1, H, W]
        cls_pred = self.cls_pred(feat)  # [B, num_anchors*1, H, W]
        
        # Reshape
        box_pred = box_pred.view(B, self.num_anchors, 4, H, W).permute(0, 1, 3, 4, 2)
        obj_pred = obj_pred.view(B, self.num_anchors, 1, H, W).permute(0, 1, 3, 4, 2)
        cls_pred = cls_pred.view(B, self.num_anchors, 1, H, W).permute(0, 1, 3, 4, 2)
        
        return {
            'boxes': box_pred,      # [B, num_anchors, H, W, 4]
            'obj': obj_pred,        # [B, num_anchors, H, W, 1]
            'cls': cls_pred         # [B, num_anchors, H, W, 1]
        }


# ==================== MAIN MODEL ====================
class FewShotYOLO(nn.Module):
    """
    Complete Few-Shot YOLO Model
    Uses pretrained YOLOv8 backbone (frozen)
    Trainable: Aggregation + Matching + Head
    """
    def __init__(self, 
                 backbone_weights: str = 'yolov8n.pt',
                 k_shot: int = 3,
                 freeze_backbone: bool = True):
        super().__init__()
        self.k_shot = k_shot
        
        # Load pretrained YOLOv8 backbone
        self.backbone = self._load_yolov8_backbone(backbone_weights)
        
        if freeze_backbone:
            self._freeze_backbone()
        
        # Few-shot components (trainable)
        self.support_aggregation = SupportAggregationModule(channels=512)
        self.matching_module = FewShotMatchingModule(channels=512)
        self.detection_head = FewShotDetectionHead(in_channels=512)
        
        print(f"✓ Model initialized with {k_shot}-shot support")
        print(f"✓ Backbone frozen: {freeze_backbone}")
        print(f"✓ Trainable params: {self._count_trainable_params():,}")
    
    def _load_yolov8_backbone(self, weights_path: str):
        """Load YOLOv8 backbone from Ultralytics"""
        try:
            from ultralytics import YOLO
            model = YOLO(weights_path)
            # Extract backbone (first 10 layers typically)
            backbone = model.model.model[:10]
            return backbone
        except:
            print("Warning: Using simple backbone (Ultralytics not available)")
            return self._create_simple_backbone()
    
    def _create_simple_backbone(self):
        """Fallback: Simple CNN backbone"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen")
    
    def _count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using backbone"""
        return self.backbone(x)
    
    def forward(self, support_imgs: torch.Tensor, query_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            support_imgs: [B, k_shot, 3, H, W]
            query_img: [B, 3, H, W]
        Returns:
            predictions: Dict with detection outputs
        """
        B, K, C, H, W = support_imgs.shape
        
        # Extract support features
        support_features = []
        for k in range(K):
            feat = self.extract_features(support_imgs[:, k])
            support_features.append(feat)
        
        # Aggregate support features
        aggregated_support = self.support_aggregation(support_features)
        
        # Extract query features
        query_features = self.extract_features(query_img)
        
        # Matching
        matched_features = self.matching_module(query_features, aggregated_support)
        
        # Detection
        predictions = self.detection_head(matched_features)
        
        return predictions


# ==================== LOSS FUNCTIONS ====================
class FewShotLoss(nn.Module):
    """
    Combined loss function with 6 components from paper
    Optimized weights for few-shot learning
    """
    def __init__(self):
        super().__init__()
        self.weights = {
            'ciou': 1.0,
            'bce': 0.8,
            'dfl': 1.2,
            'focal': 1.5,
            'rpl': 2.0,      # Highest for few-shot
            'dice': 1.0
        }
    
    def compute_ciou_loss(self, pred_boxes, target_boxes):
        """Complete IoU Loss"""
        # Simplified version - implement full CIoU as needed
        iou = self._box_iou(pred_boxes, target_boxes)
        return 1 - iou.mean()
    
    def compute_bce_loss(self, pred, target):
        """Binary Cross Entropy"""
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    
    def compute_focal_loss(self, pred, target, gamma=1.5, alpha=0.25):
        """Focal Loss (from paper: γ=1.5, α=0.25)"""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def compute_rpl_loss(self, pred, target):
        """Ratio-Preserving Loss (critical for few-shot)"""
        pos_mask = target > 0.5
        neg_mask = ~pos_mask
        
        num_pos = pos_mask.sum() + 1e-6
        num_neg = neg_mask.sum() + 1e-6
        
        pos_loss = F.binary_cross_entropy_with_logits(pred[pos_mask], target[pos_mask], reduction='sum')
        neg_loss = F.binary_cross_entropy_with_logits(pred[neg_mask], target[neg_mask], reduction='sum')
        
        # Balance positive and negative
        ratio = num_neg / num_pos
        rpl = (pos_loss * ratio + neg_loss) / (num_pos + num_neg)
        return rpl
    
    def compute_dice_loss(self, pred, target):
        """Dice Loss for overlap optimization"""
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice
    
    def _box_iou(self, boxes1, boxes2):
        """Compute IoU between boxes"""
        # Simplified - implement full IoU as needed
        return torch.rand(boxes1.shape[0]).to(boxes1.device)  # Placeholder
    
    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        Returns dict with individual losses for logging
        """
        # Extract predictions
        pred_boxes = predictions['boxes']
        pred_obj = predictions['obj']
        pred_cls = predictions['cls']
        
        target_boxes = targets['boxes']
        target_obj = targets['obj']
        target_cls = targets['cls']
        
        # Compute individual losses
        loss_ciou = self.compute_ciou_loss(pred_boxes, target_boxes)
        loss_bce = self.compute_bce_loss(pred_obj, target_obj)
        loss_focal = self.compute_focal_loss(pred_cls, target_cls)
        loss_rpl = self.compute_rpl_loss(pred_obj, target_obj)
        loss_dice = self.compute_dice_loss(pred_cls, target_cls)
        
        # DFL placeholder
        loss_dfl = torch.tensor(0.0).to(pred_boxes.device)
        
        # Total loss
        total_loss = (
            self.weights['ciou'] * loss_ciou +
            self.weights['bce'] * loss_bce +
            self.weights['dfl'] * loss_dfl +
            self.weights['focal'] * loss_focal +
            self.weights['rpl'] * loss_rpl +
            self.weights['dice'] * loss_dice
        )
        
        return {
            'total': total_loss,
            'ciou': loss_ciou,
            'bce': loss_bce,
            'focal': loss_focal,
            'rpl': loss_rpl,
            'dice': loss_dice
        }


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
        # Simplified - expand as needed
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0),
            'obj': torch.ones(1, dtype=torch.float32),  # Placeholder
            'cls': torch.ones(1, dtype=torch.float32)   # Placeholder
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
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Loss and optimizer
        self.criterion = FewShotLoss()
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
            query_target = {k: v.to(self.device) for k, v in batch['query_target'].items()}
            
            # Forward
            self.optimizer.zero_grad()
            predictions = self.model(support_imgs, query_img)
            
            # Compute loss
            losses = self.criterion(predictions, query_target)
            loss = losses['total']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
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
                query_target = {k: v.to(self.device) for k, v in batch['query_target'].items()}
                
                predictions = self.model(support_imgs, query_img)
                losses = self.criterion(predictions, query_target)
                
                total_loss += losses['total'].item()
        
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
            predictions = self.model(support_batch, query_tensor)
        
        # Post-process predictions
        boxes, scores, class_ids = self._postprocess(predictions, query_img_orig.shape[:2])
        
        # Visualize
        vis_img = self._visualize(query_img_orig, boxes, scores)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'visualization': vis_img
        }
    
    def _postprocess(self, predictions: Dict, orig_shape: Tuple[int, int]):
        """
        Post-process predictions: NMS, coordinate conversion, etc.
        
        Args:
            predictions: Model output dict
            orig_shape: Original image shape (H, W)
            
        Returns:
            boxes: [N, 4] in xyxy format
            scores: [N]
            class_ids: [N]
        """
        pred_boxes = predictions['boxes']  # [B, num_anchors, H, W, 4]
        pred_obj = predictions['obj']      # [B, num_anchors, H, W, 1]
        pred_cls = predictions['cls']      # [B, num_anchors, H, W, 1]
        
        # Flatten predictions
        B, A, H, W, _ = pred_boxes.shape
        pred_boxes = pred_boxes.view(B, -1, 4)  # [B, A*H*W, 4]
        pred_obj = pred_obj.view(B, -1)         # [B, A*H*W]
        pred_cls = pred_cls.view(B, -1)         # [B, A*H*W]
        
        # Apply sigmoid
        pred_obj = torch.sigmoid(pred_obj)
        pred_cls = torch.sigmoid(pred_cls)
        
        # Compute scores
        scores = pred_obj * pred_cls  # [B, A*H*W]
        
        # Filter by confidence
        mask = scores[0] > self.conf_threshold
        boxes = pred_boxes[0][mask]
        scores = scores[0][mask]
        
        # Convert to xyxy format and scale to original size
        boxes = self._xywh_to_xyxy(boxes)
        boxes = self._scale_boxes(boxes, (640, 640), orig_shape)
        
        # NMS
        keep_indices = self._nms(boxes, scores, self.iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = torch.zeros(len(boxes), dtype=torch.long)  # Single-class
        
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
        'data_root': 'dataset/',
        'n_way': 5,
        'k_shot': 3,
        'n_query': 5,
        'img_size': 640,
        
        # Model
        'backbone_weights': 'yolov8n.pt',
        'freeze_backbone': True,
        
        # Training
        'batch_size': 4,
        'lr': 1e-4,
        'epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Paths
        'save_dir': 'checkpoints/',
        'results_dir': 'results/'
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