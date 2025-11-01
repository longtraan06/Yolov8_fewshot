# 🎯 Few-Shot YOLO - Complete Usage Guide

## 📋 Table of Contents
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Inference](#inference)
5. [Model Architecture](#model-architecture)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Requirements
```bash
Python >= 3.8
PyTorch >= 1.12
CUDA >= 11.3 (recommended)
```

### Step 1: Install Dependencies
```bash
# Create virtual environment
conda create -n fewshot_yolo python=3.9
conda activate fewshot_yolo

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install ultralytics
pip install opencv-python
pip install albumentations
pip install tqdm
pip install pyyaml
pip install matplotlib
pip install seaborn
```

### Step 2: Download YOLOv8 Weights
```bash
# Download pretrained YOLOv8 backbone
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Step 3: Verify Installation
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

## 📁 Dataset Preparation

### Dataset Structure

Your dataset must follow this **exact structure**:

```
dataset/
├── train/
│   ├── class_1/
│   │   ├── support/
│   │   │   ├── 001.jpg
│   │   │   ├── 001.txt
│   │   │   ├── 002.jpg
│   │   │   ├── 002.txt
│   │   │   └── ...
│   │   └── query/
│   │       ├── 001.jpg
│   │       ├── 001.txt
│   │       └── ...
│   ├── class_2/
│   │   ├── support/
│   │   └── query/
│   └── ...
│
├── val/
│   └── (same structure as train)
│
└── test/
    └── (same structure as train)
```

### Annotation Format (YOLO Format)

Each `.txt` file contains bounding boxes in YOLO format:
```
class_id x_center y_center width height
```

**Example `001.txt`:**
```
0 0.5 0.5 0.3 0.4
0 0.2 0.3 0.15 0.2
```

- All values are **normalized** to [0, 1]
- `class_id`: 0 (single-class detection)
- `x_center, y_center`: Box center coordinates
- `width, height`: Box dimensions

### Dataset Requirements

| Split | Min Classes | Min Images/Class | Purpose |
|-------|-------------|------------------|---------|
| **Train** | 20-30 | 20-30 | Training episodes |
| **Val** | 10-15 | 15-20 | Validation |
| **Test** | 10-15 | 15-20 | Novel class testing |

### Creating Dataset from COCO

```python
import json
import shutil
from pathlib import Path

def convert_coco_to_fewshot(coco_json, images_dir, output_dir):
    """Convert COCO format to Few-Shot format"""
    
    with open(coco_json, 'r') as f:
        coco = json.load(f)
    
    # Group images by category
    category_images = {}
    for ann in coco['annotations']:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        
        if cat_id not in category_images:
            category_images[cat_id] = []
        category_images[cat_id].append((img_id, ann))
    
    # Create dataset structure
    for cat_id, items in category_images.items():
        cat_name = f"class_{cat_id}"
        
        # Split into support and query
        support_items = items[:5]  # First 5 as support
        query_items = items[5:]    # Rest as query
        
        # Create directories
        support_dir = Path(output_dir) / 'train' / cat_name / 'support'
        query_dir = Path(output_dir) / 'train' / cat_name / 'query'
        support_dir.mkdir(parents=True, exist_ok=True)
        query_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and convert support images
        for idx, (img_id, ann) in enumerate(support_items):
            img_info = next(img for img in coco['images'] if img['id'] == img_id)
            src_img = Path(images_dir) / img_info['file_name']
            dst_img = support_dir / f"{idx:03d}.jpg"
            shutil.copy(src_img, dst_img)
            
            # Create YOLO annotation
            create_yolo_annotation(ann, img_info, support_dir / f"{idx:03d}.txt")
        
        # Similarly for query images
        for idx, (img_id, ann) in enumerate(query_items):
            # ... (same process)
            pass

def create_yolo_annotation(ann, img_info, output_path):
    """Convert COCO bbox to YOLO format"""
    x, y, w, h = ann['bbox']
    img_w, img_h = img_info['width'], img_info['height']
    
    # Convert to YOLO format
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    
    with open(output_path, 'w') as f:
        f.write(f"0 {x_center} {y_center} {width} {height}\n")

# Usage
convert_coco_to_fewshot(
    coco_json='coco/annotations/instances_train2017.json',
    images_dir='coco/train2017',
    output_dir='dataset/'
)
```

### Quick Dataset Creation Script

```python
import os
import random
import shutil
from pathlib import Path

def create_sample_dataset(output_dir='dataset_sample', num_classes=5, images_per_class=20):
    """Create a sample dataset structure for testing"""
    
    output_path = Path(output_dir)
    
    for split in ['train', 'val', 'test']:
        for class_id in range(num_classes):
            class_name = f"class_{class_id}"
            
            # Create directories
            support_dir = output_path / split / class_name / 'support'
            query_dir = output_path / split / class_name / 'query'
            support_dir.mkdir(parents=True, exist_ok=True)
            query_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images (replace with your actual images)
            for i in range(5):  # 5 support images
                img_path = support_dir / f"{i:03d}.jpg"
                txt_path = support_dir / f"{i:03d}.txt"
                
                # Create dummy image (replace with actual image copy)
                # shutil.copy(source_img, img_path)
                
                # Create annotation
                with open(txt_path, 'w') as f:
                    x, y, w, h = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7), 0.2, 0.2
                    f.write(f"0 {x} {y} {w} {h}\n")
            
            # Create query images
            for i in range(images_per_class - 5):
                img_path = query_dir / f"{i:03d}.jpg"
                txt_path = query_dir / f"{i:03d}.txt"
                
                # Similar process for query images
                # ...
    
    print(f"✓ Sample dataset created at {output_dir}")

# Create sample dataset
create_sample_dataset()
```

---

## 🚀 Training

### Step 1: Prepare Configuration

Edit the `CONFIG` dict in the main script:

```python
CONFIG = {
    # Data paths
    'data_root': 'dataset/',          # Your dataset root
    
    # Episode sampling
    'n_way': 5,                       # Number of classes per episode
    'k_shot': 3,                      # Number of support images (IMPORTANT!)
    'n_query': 5,                     # Number of query images per class
    'img_size': 640,                  # Image size
    
    # Model
    'backbone_weights': 'yolov8n.pt', # Pretrained backbone
    'freeze_backbone': True,          # Freeze backbone (RECOMMENDED!)
    
    # Training hyperparameters
    'batch_size': 4,                  # Batch size (adjust based on GPU)
    'lr': 1e-4,                       # Learning rate
    'epochs': 100,                    # Number of epochs
    'device': 'cuda',                 # 'cuda' or 'cpu'
    
    # Output
    'save_dir': 'checkpoints/',       # Model checkpoints
    'results_dir': 'results/'         # Results and visualizations
}
```

### Step 2: Run Training

```bash
# Basic training
python fewshot_yolo.py

# Training with custom config
python fewshot_yolo.py --config config.yaml

# Resume from checkpoint
python fewshot_yolo.py --resume checkpoints/checkpoint_epoch_50.pth
```

### Step 3: Monitor Training

Training logs will show:
```
Epoch 1/100
Train Loss: 2.3456
Val Loss: 2.1234
LR: 0.000100
------------------------------------------------------------
✓ Best model saved at epoch 1
```

### Training Tips

#### 1. **GPU Memory Management**
```python
# If out of memory, reduce batch size
'batch_size': 2  # Instead of 4

# Or use gradient accumulation
accumulation_steps = 2
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. **Learning Rate Tuning**
```python
# Start with conservative LR for frozen backbone
'lr': 1e-4  # Default

# If training is unstable
'lr': 5e-5  # Lower

# If training is too slow
'lr': 2e-4  # Higher
```

#### 3. **Checkpoint Strategy**
```python
# Save every 5 epochs + best model
if epoch % 5 == 0 or is_best:
    save_checkpoint(epoch, is_best)
```

#### 4. **Data Augmentation Control**
```python
# For small datasets, use heavier augmentation
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),           # Add vertical flip
    A.RandomRotate90(p=0.3),         # Add rotation
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
])
```

---

## 🔍 Inference

### Basic Inference

```python
from fewshot_yolo import FewShotYOLO, FewShotInference

# Initialize model
model = FewShotYOLO(
    backbone_weights='yolov8n.pt',
    k_shot=3,
    freeze_backbone=True
)

# Load trained weights
inference = FewShotInference(
    model=model,
    checkpoint_path='checkpoints/best_model.pth',
    device='cuda',
    conf_threshold=0.5,
    iou_threshold=0.45
)

# Prepare images
support_images = [
    'examples/support_1.jpg',
    'examples/support_2.jpg',
    'examples/support_3.jpg'
]
query_image = 'examples/query.jpg'

# Run detection
results = inference.detect(support_images, query_image)

# Save results
inference.save_results(results, 'output/result.jpg')
```

### Batch Inference

```python
import glob

# Get all query images
query_images = glob.glob('test_images/*.jpg')

# Fixed support set
support_images = [
    'support/example_1.jpg',
    'support/example_2.jpg',
    'support/example_3.jpg'
]

# Process all queries
for query_path in query_images:
    results = inference.detect(support_images, query_path)
    
    output_path = f"results/{Path(query_path).stem}_result.jpg"
    inference.save_results(results, output_path)
    
    print(f"✓ Processed {query_path}: {len(results['boxes'])} objects detected")
```

### Custom Post-Processing

```python
def custom_postprocess(results, min_score=0.7, min_area=100):
    """Custom filtering of detection results"""
    
    boxes = results['boxes']
    scores = results['scores']
    
    # Filter by score
    mask = scores > min_score
    boxes = boxes[mask]
    scores = scores[mask]
    
    # Filter by area
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    mask = areas > min_area
    boxes = boxes[mask]
    scores = scores[mask]
    
    return boxes, scores

# Usage
results = inference.detect(support_images, query_image)
filtered_boxes, filtered_scores = custom_postprocess(results)
```

### Visualization Options

```python
import cv2
import matplotlib.pyplot as plt

def visualize_detections(img_path, boxes, scores, save_path=None):
    """Enhanced visualization"""
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        
        # Draw box
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        plt.gca().add_patch(rect)
        
        # Draw label
        plt.text(
            x1, y1-10,
            f'Score: {score:.2f}',
            bbox=dict(facecolor='red', alpha=0.5),
            fontsize=10,
            color='white'
        )
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

# Usage
results = inference.detect(support_images, query_image)
visualize_detections(
    query_image,
    results['boxes'],
    results['scores'],
    save_path='output/visualization.png'
)
```

---

## 🏗️ Model Architecture

### Overview

```
Input: Support Images [K, 3, 640, 640] + Query Image [3, 640, 640]
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
[Support Branch]               [Query Branch]
    │                                 │
    ▼                                 ▼
YOLOv8 Backbone (FROZEN)      YOLOv8 Backbone (FROZEN)
    │                                 │
    │ [Features: K x [C, H, W]]       │ [Features: [C, H, W]]
    ▼                                 │
Support Aggregation Module            │
    │                                 │
    │ [Aggregated: [C, H, W]]         │
    └──────────────┬──────────────────┘
                   │
                   ▼
           Matching Module
                   │
                   ▼
           Detection Head
                   │
                   ▼
        [Boxes, Scores, Classes]
```

### Component Details

#### 1. **Support Aggregation Module**
- **Input**: K feature maps from K support images
- **Process**:
  - Level 1: Spatial Attention → Focus on object regions
  - Level 2: Channel Attention → Emphasize discriminative channels
  - Level 3: Refinement → Residual learning
- **Output**: Single aggregated prototype
- **Params**: ~500K (< 5% of total)

#### 2. **Matching Module**
- **Input**: Query features + Aggregated support
- **Process**:
  - Corr_1: Element-wise multiplication
  - Corr_2: Spatial correlation (Conv3x3)
  - Corr_3: Global correlation
  - Fusion + Adaptive weighting
- **Output**: Matched features
- **Params**: ~300K

#### 3. **Detection Head**
- **Input**: Matched features
- **Output**: Bounding boxes + Objectness + Class scores
- **Params**: ~200K

**Total Trainable Parameters**: ~1M (YOLOv8 backbone frozen)

### Loss Function Weights

```python
loss_weights = {
    'ciou': 1.0,    # Box regression
    'bce': 0.8,     # Binary classification
    'dfl': 1.2,     # Distribution focal
    'focal': 1.5,   # Hard example mining
    'rpl': 2.0,     # **CRITICAL** for few-shot (imbalance handling)
    'dice': 1.0     # Overlap optimization
}
```

**Why RPL has highest weight?**
- Few-shot → Very few positive samples (only 3!)
- RPL adaptively balances positive/negative samples
- Prevents model from being overwhelmed by negatives

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
```python
# Solution 1: Reduce batch size
'batch_size': 2  # or even 1

# Solution 2: Reduce image size
'img_size': 512  # instead of 640

# Solution 3: Use gradient checkpointing
model.gradient_checkpointing = True

# Solution 4: Clear cache
import torch
torch.cuda.empty_cache()
```

### Issue 2: Training Loss Not Decreasing

**Symptoms:**
- Loss stays constant or increases
- Validation loss much higher than training loss

**Solutions:**
```python
# Check 1: Learning rate too high
'lr': 5e-5  # Reduce from 1e-4

# Check 2: Verify data loading
for batch in dataloader:
    print(batch['support_images'].shape)  # Should be [B, K, 3, H, W]
    print(batch['query_image'].shape)     # Should be [B, 3, H, W]
    break

# Check 3: Verify labels
print("Support labels:", batch['support_labels'])
print("Query labels:", batch['query_labels'])

# Check 4: Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

### Issue 3: Poor Detection Results

**Symptoms:**
- No detections or very low confidence scores
- Many false positives

**Solutions:**
```python
# Adjust confidence threshold
inference = FewShotInference(
    model=model,
    checkpoint_path='best_model.pth',
    conf_threshold=0.3,  # Lower from 0.5
    iou_threshold=0.45
)

# Check support image quality
# - Support images should clearly show the object
# - Support images should have good diversity
# - Support images should be well-annotated

# Verify model learned something
print("Model predictions on training data:")
model.eval()
with torch.no_grad():
    preds = model(train_support, train_query)
    print("Objectness:", preds['obj'].mean())
    print("Class scores:", preds['cls'].mean())
```

### Issue 4: Backbone Loading Failed

**Symptoms:**
```
Error: Cannot load YOLOv8 backbone
```

**Solutions:**
```python
# Solution 1: Install ultralytics properly
pip install ultralytics==8.0.0

# Solution 2: Download weights manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Solution 3: Use fallback backbone
model = FewShotYOLO(
    backbone_weights=None,  # Will use simple CNN backbone
    k_shot=3
)
```

### Issue 5: Episode Sampling Error

**Symptoms:**
```
IndexError: Cannot sample N classes from M available
```

**Solutions:**
```python
# Check dataset structure
print("Available classes:", len(dataset.classes))
print("Required for N-way:", CONFIG['n_way'])

# Reduce n_way if needed
CONFIG['n_way'] = min(5, len(dataset.classes))

# Or add more classes to dataset
```

---

## 📊 Expected Performance

### Training Metrics

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **Train Loss** | 1.5 - 2.5 | Should decrease steadily |
| **Val Loss** | 1.8 - 3.0 | Should track train loss |
| **mAP@0.5** | 0.4 - 0.7 | Depends on dataset |
| **Training Time** | 2-5 hours | For 100 epochs on V100 |

### Inference Speed

| Device | FPS | Latency |
|--------|-----|---------|
| **RTX 3090** | ~30-40 | ~25ms |
| **V100** | ~25-35 | ~30ms |
| **RTX 2080 Ti** | ~20-25 | ~40ms |
| **CPU (16 cores)** | ~2-3 | ~400ms |

---

## 📚 Advanced Usage

### Custom Backbone

```python
class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom backbone
        self.features = nn.Sequential(...)
    
    def forward(self, x):
        return self.features(x)

# Use custom backbone
model = FewShotYOLO(
    backbone_weights=None,
    k_shot=3
)
model.backbone = CustomBackbone()
```

### Multi-GPU Training

```python
# Wrap model with DataParallel
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# Or use DistributedDataParallel (faster)
import torch.distributed as dist
model = nn.parallel.DistributedDataParallel(model)
```

### Export to ONNX

```python
# Export model
dummy_support = torch.randn(1, 3, 3, 640, 640)
dummy_query = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    (dummy_support, dummy_query),
    'fewshot_yolo.onnx',
    opset_version=11,
    input_names=['support', 'query'],
    output_names=['boxes', 'scores']
)
```

---

## 📞 Support

For issues and questions:
1. Check this guide first
2. Review the code comments
3. Check GitHub issues
4. Contact: [your email]

---

**Happy Training! 🚀**