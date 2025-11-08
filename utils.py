import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import json
import shutil

class MetricsCalculator:
    """Calculate detection metrics (mAP, precision, recall)"""
    
    @staticmethod
    def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between two sets of boxes
        boxes: [N, 4] in xyxy format
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-6)
        
        return iou
    
    @staticmethod
    def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Calculate Average Precision"""
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        
        return ap
    
    @staticmethod
    def evaluate_detections(pred_boxes: List[np.ndarray],
                          pred_scores: List[np.ndarray],
                          gt_boxes: List[np.ndarray],
                          iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate detections across all images
        
        Args:
            pred_boxes: List of predicted boxes per image [N, 4]
            pred_scores: List of prediction scores per image [N]
            gt_boxes: List of ground truth boxes per image [M, 4]
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            metrics: Dict with precision, recall, F1, mAP
        """
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        
        # Combine all predictions and ground truths
        for pred_box, pred_score, gt_box in zip(pred_boxes, pred_scores, gt_boxes):
            all_pred_boxes.extend(pred_box)
            all_pred_scores.extend(pred_score)
            all_gt_boxes.extend([gt_box] * len(pred_box))
        
        if len(all_pred_boxes) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mAP': 0.0}
        
        # Sort by confidence
        indices = np.argsort(all_pred_scores)[::-1]
        all_pred_boxes = np.array(all_pred_boxes)[indices]
        all_pred_scores = np.array(all_pred_scores)[indices]
        
        # Calculate TP, FP, FN
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        
        num_gt = sum(len(gt) for gt in gt_boxes)
        
        for i, (pred_box, gt_box_set) in enumerate(zip(all_pred_boxes, all_gt_boxes)):
            if len(gt_box_set) == 0:
                fp[i] = 1
                continue
            
            ious = MetricsCalculator.box_iou(
                pred_box[None, :],
                gt_box_set
            )[0]
            
            max_iou = ious.max()
            if max_iou >= iou_threshold:
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Calculate metrics
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / (num_gt + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Calculate AP
        ap = MetricsCalculator.calculate_ap(recalls, precisions)
        
        # Final metrics
        precision = precisions[-1] if len(precisions) > 0 else 0.0
        recall = recalls[-1] if len(recalls) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'mAP': float(ap)
        }


class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def draw_boxes(image: np.ndarray,
                   boxes: np.ndarray,
                   scores: np.ndarray = None,
                   labels: List[str] = None,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image [H, W, 3]
            boxes: Boxes [N, 4] in xyxy format
            scores: Confidence scores [N]
            labels: Class labels [N]
            color: Box color (B, G, R)
            thickness: Box thickness
            
        Returns:
            image_with_boxes: Annotated image
        """
        img = image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label_text = ""
            if labels is not None:
                label_text = labels[i]
            if scores is not None:
                label_text += f" {scores[i]:.2f}"
            
            if label_text:
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1
                )
                
                # Text
                cv2.putText(
                    img, label_text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
        
        return img
    
    @staticmethod
    def visualize_support_query(support_images: List[np.ndarray],
                               query_image: np.ndarray,
                               query_boxes: np.ndarray = None,
                               save_path: str = None):
        """
        Visualize support set and query with detections
        
        Args:
            support_images: List of K support images
            query_image: Query image
            query_boxes: Detected boxes on query
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        k_shot = len(support_images)
        
        fig, axes = plt.subplots(1, k_shot + 1, figsize=(4 * (k_shot + 1), 4))
        
        # Plot support images
        for i, (ax, img) in enumerate(zip(axes[:k_shot], support_images)):
            ax.imshow(img)
            ax.set_title(f'Support {i+1}')
            ax.axis('off')
        
        # Plot query with detections
        query_vis = query_image.copy()
        if query_boxes is not None:
            query_vis = Visualizer.draw_boxes(query_vis, query_boxes)
        
        axes[k_shot].imshow(query_vis)
        axes[k_shot].set_title(f'Query ({len(query_boxes)} detections)')
        axes[k_shot].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class DatasetConverter:
    """Convert various dataset formats to Few-Shot format"""
    
    @staticmethod
    def coco_to_fewshot(coco_json: str,
                       images_dir: str,
                       output_dir: str,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       support_per_class: int = 5):
        """
        Convert COCO format to Few-Shot format
        
        Args:
            coco_json: Path to COCO annotations JSON
            images_dir: Directory containing images
            output_dir: Output directory
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            support_per_class: Number of support images per class
        """
        import json
        from collections import defaultdict
        
        with open(coco_json, 'r') as f:
            coco = json.load(f)
        
        # Build image info mapping
        img_info = {img['id']: img for img in coco['images']}
        
        # Group annotations by category
        cat_anns = defaultdict(list)
        for ann in coco['annotations']:
            cat_anns[ann['category_id']].append(ann)
        
        # Process each category
        for cat_id, anns in cat_anns.items():
            # Get category name
            cat_name = next(
                cat['name'] for cat in coco['categories'] if cat['id'] == cat_id
            )
            cat_name = cat_name.replace(' ', '_')
            
            # Shuffle annotations
            np.random.shuffle(anns)
            
            # Split into train/val/test
            n = len(anns)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits = {
                'train': anns[:train_end],
                'val': anns[train_end:val_end],
                'test': anns[val_end:]
            }
            
            # Create directory structure
            for split, split_anns in splits.items():
                support_dir = Path(output_dir) / split / cat_name / 'support'
                query_dir = Path(output_dir) / split / cat_name / 'query'
                support_dir.mkdir(parents=True, exist_ok=True)
                query_dir.mkdir(parents=True, exist_ok=True)
                
                # Split into support and query
                support_anns = split_anns[:support_per_class]
                query_anns = split_anns[support_per_class:]
                
                # Save support images
                DatasetConverter._save_coco_images(
                    support_anns, img_info, images_dir, support_dir
                )
                
                # Save query images
                DatasetConverter._save_coco_images(
                    query_anns, img_info, images_dir, query_dir
                )
        
        print(f"✓ Converted COCO dataset to {output_dir}")
    
    @staticmethod
    def _save_coco_images(anns: List[Dict],
                         img_info: Dict,
                         images_dir: str,
                         output_dir: Path):
        """Helper to save COCO images and annotations"""
        for idx, ann in enumerate(anns):
            img_id = ann['image_id']
            img_data = img_info[img_id]
            
            # Copy image
            src_path = Path(images_dir) / img_data['file_name']
            dst_img = output_dir / f"{idx:04d}.jpg"
            
            if src_path.exists():
                shutil.copy(src_path, dst_img)
            
            # Create YOLO annotation
            bbox = ann['bbox']  # [x, y, w, h]
            img_w, img_h = img_data['width'], img_data['height']
            
            # Convert to YOLO format
            x_center = (bbox[0] + bbox[2] / 2) / img_w
            y_center = (bbox[1] + bbox[3] / 2) / img_h
            width = bbox[2] / img_w
            height = bbox[3] / img_h
            
            dst_txt = output_dir / f"{idx:04d}.txt"
            with open(dst_txt, 'w') as f:
                f.write(f"0 {x_center} {y_center} {width} {height}\n")


class ConfigManager:
    """Manage training configurations"""
    
    @staticmethod
    def save_config(config: Dict, save_path: str):
        """Save config to YAML"""
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ Config saved to {save_path}")
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load config from YAML"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default configuration"""
        return {
            'data': {
                'data_root': 'dataset/',
                'n_way': 5,
                'k_shot': 3,
                'n_query': 5,
                'img_size': 640,
            },
            'model': {
                'backbone_weights': 'yolov8n.pt',
                'freeze_backbone': True,
            },
            'training': {
                'batch_size': 4,
                'lr': 1e-4,
                'epochs': 100,
                'device': 'cuda',
                'num_workers': 4,
            },
            'loss': {
                'ciou': 1.0,
                'bce': 0.8,
                'dfl': 1.2,
                'focal': 1.5,
                'rpl': 2.0,
                'dice': 1.0,
            },
            'paths': {
                'save_dir': 'checkpoints/',
                'results_dir': 'results/',
                'logs_dir': 'logs/',
            }
        }


class CheckpointManager:
    """Manage model checkpoints"""
    
    @staticmethod
    def save_checkpoint(state: Dict, save_path: str, is_best: bool = False):
        """Save checkpoint"""
        torch.save(state, save_path)
        
        if is_best:
            best_path = Path(save_path).parent / 'best_model.pth'
            shutil.copy(save_path, best_path)
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer=None):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('history', {})


# ==================== EXAMPLE USAGE ====================
if __name__ == '__main__':
    # Example 1: Calculate metrics
    print("=" * 60)
    print("Example 1: Calculate Detection Metrics")
    print("=" * 60)
    
    # Dummy predictions and ground truths
    pred_boxes = [np.array([[100, 100, 200, 200], [300, 300, 400, 400]])]
    pred_scores = [np.array([0.9, 0.7])]
    gt_boxes = [np.array([[105, 105, 205, 205], [310, 310, 410, 410]])]
    
    metrics = MetricsCalculator.evaluate_detections(
        pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5
    )
    
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    
    # Example 2: Create default config
    print("\n" + "=" * 60)
    print("Example 2: Create Default Config")
    print("=" * 60)
    
    config = ConfigManager.get_default_config()
    ConfigManager.save_config(config, 'config.yaml')
    
    # Example 3: Load and modify config
    loaded_config = ConfigManager.load_config('config.yaml')
    loaded_config['training']['epochs'] = 200
    print(f"Modified epochs: {loaded_config['training']['epochs']}")
    
    print("\n✓ All examples completed!")


# docker run -itd \
#   --gpus all \
#   --network host \
#   --shm-size=6g \
#   -v /mlcv2/WorkingSpace/Personal/chinhnm:/mlcv2/WorkingSpace/Personal/chinhnm \
#   --name fewshot \
#   --restart unless-stopped \
#   chinhcachep:latest bash
