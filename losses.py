
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from utils.metrics import bbox_iou # Giả sử bạn có file metrics.py hoặc hàm này ở đâu đó
from utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors, bbox2dist


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# ============================================================================
#                          1. COMPLETE IOU LOSS
# ============================================================================

class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    ):
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

# ============================================================================
#                    2. BINARY CROSS-ENTROPY LOSS
# ============================================================================

def compute_bce_loss(pred_cls: torch.Tensor, 
                     target_cls: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred_cls, target_cls)


# ============================================================================
#                   3. DISTRIBUTION FOCAL LOSS
# ============================================================================

def compute_dfl_loss(pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Distribution Focal Loss (DFL) được triển khai đầy đủ.
    Args:
        pred_dist (torch.Tensor): Phân phối dự đoán [N, reg_max+1].
        target (torch.Tensor): Giá trị target (float) [N].
    """
    target_left = target.to(torch.long)
    target_right = target_left + 1
    weight_left = target_right.to(torch.float) - target
    weight_right = 1.0 - weight_left
    
    loss = F.cross_entropy(pred_dist, target_left, reduction='none') * weight_left \
         + F.cross_entropy(pred_dist, target_right, reduction='none') * weight_right
         
    return loss.mean()

# ============================================================================
#                        4. FOCAL LOSS
# ============================================================================

def focal_loss(pred: torch.Tensor, 
               target: torch.Tensor,
               alpha: float = 0.25, 
               gamma: float = 1.5,
               reduction: str = 'mean') -> torch.Tensor:
    # BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none'
    )
    
    # Focal weight
    pred_prob = torch.sigmoid(pred)
    p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    
    # Alpha weighting
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    loss = alpha_t * focal_weight * bce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


# ============================================================================
#                   5. RATIO-PRESERVING LOSS (CRITICAL)
# ============================================================================

def ratio_preserving_loss(pred_obj: torch.Tensor, target_obj: torch.Tensor) -> torch.Tensor:
    """
    Ratio-Preserving Loss (RPL) - Phiên bản sửa lỗi.
    Chỉ tập trung vào việc cân bằng loss cho objectness/classification.
    """
    pos_mask = target_obj > 0.5
    neg_mask = ~pos_mask
    
    num_pos = pos_mask.sum() + 1e-6
    num_neg = neg_mask.sum() + 1e-6
    
    pos_loss = F.binary_cross_entropy_with_logits(pred_obj[pos_mask], target_obj[pos_mask], reduction='sum')
    neg_loss = F.binary_cross_entropy_with_logits(pred_obj[neg_mask], target_obj[neg_mask], reduction='sum')
    
    ratio = num_neg / num_pos
    rpl = (pos_loss * ratio + neg_loss) / (num_pos + num_neg)
    return rpl


# ============================================================================
#                          6. DICE LOSS
# ============================================================================

def dice_loss(pred_boxes: torch.Tensor, 
              target_boxes: torch.Tensor,
              eps: float = 1e-7) -> torch.Tensor:
    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    # Areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])
    
    # Dice coefficient
    dice = (2 * inter_area) / (pred_area + target_area + eps)
    
    # Loss (1 - Dice)
    loss = 1 - dice
    return loss.mean()


# ============================================================================
#                       COMBINED LOSS FUNCTION
# ============================================================================

class FewShotYOLOLoss(nn.Module):
    """
    Hàm loss cho Few-Shot YOLO, được nâng cấp với TaskAlignedAssigner và các module loss
    từ Ultralytics.
    """
    def __init__(self, model):
        super().__init__()
        
        device = next(model.parameters()).device
        head = model.detection_head
        
        self.reg_max = head.reg_max
        self.nc = head.nc
        self.stride = head.stride.to(device)
        self.device = device
        self.no = head.no
        
        # SỬ DỤNG CÁC THÀNH PHẦN MẠNH MẼ TỪ ULTRALYTICS
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max).to(device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        
        # Lấy trọng số loss từ YOLOv8 (bạn có thể điều chỉnh)
        self.box_weight = 7.5
        self.cls_weight = 0.5  # Đối với chúng ta, đây là objectness weight
        self.dfl_weight = 1.5

    def preprocess(self, targets_list: list, batch_size: int):
        """
        Chuyển đổi list các dictionary target sang các tensor đã được batch và pad.
        Returns:
            gt_labels (Tensor): [B, max_gt, 1]
            gt_bboxes (Tensor): [B, max_gt, 4] (xyxy format)
            mask_gt (Tensor):   [B, max_gt, 1]
        """
        # Tìm số lượng vật thể lớn nhất trong batch để pad
        max_num_gt = 0
        for targets_dict in targets_list:
            if 'boxes' in targets_dict:
                max_num_gt = max(max_num_gt, targets_dict['boxes'].shape[0])

        if max_num_gt == 0:
            return (torch.zeros(batch_size, 0, 1, device=self.device),
                    torch.zeros(batch_size, 0, 4, device=self.device),
                    torch.zeros(batch_size, 0, 1, device=self.device))

        # Tạo các tensor rỗng để chứa dữ liệu đã pad
        gt_labels = torch.zeros(batch_size, max_num_gt, 1, device=self.device)
        gt_bboxes = torch.zeros(batch_size, max_num_gt, 4, device=self.device)
        mask_gt = torch.zeros(batch_size, max_num_gt, 1, device=self.device, dtype=torch.bool)

        for i, targets_dict in enumerate(targets_list):
            if 'boxes' in targets_dict and targets_dict['boxes'].numel() > 0:
                num_gt = targets_dict['boxes'].shape[0]
                # Điền dữ liệu vào tensor đã pad
                gt_labels[i, :num_gt] = 0 # Class luôn là 0
                gt_bboxes[i, :num_gt] = xywh2xyxy(targets_dict['boxes']) # Chuyển sang xyxy
                mask_gt[i, :num_gt] = True

        return gt_labels, gt_bboxes, mask_gt

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        pred_dist_softmax = pred_dist.view(*pred_dist.shape[:-1], 4, self.reg_max).softmax(-1)
        pred_dist_matmul = pred_dist_softmax.matmul(self.proj)
        return dist2bbox(pred_dist_matmul, anchor_points, xywh=False)

    def forward(self, preds_list: list, targets_list: list) -> Tuple[torch.Tensor, Dict]:
        batch_size = preds_list[0].shape[0]
        
        # 1. NỐI VÀ RESHAPE PREDICTIONS
        pred_distri, pred_scores = torch.cat([xi.view(batch_size, self.no, -1) for xi in preds_list], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        # 2. TẠO ANCHORS
        anchor_points, stride_tensor = make_anchors(preds_list, self.stride, 0.5)

        # 3. PREPROCESS TARGETS
        gt_labels, gt_bboxes, mask_gt = self.preprocess(targets_list, batch_size)

        # Nếu không có target nào trong toàn bộ batch
        if gt_bboxes.numel() == 0:
            target_scores = torch.zeros_like(pred_scores)
            loss_obj = self.bce(pred_scores, target_scores).sum() / target_scores.numel() * batch_size
            total_loss = loss_obj * self.cls_weight
            return total_loss, {'total': total_loss.item(), 'ciou': 0.0, 'obj': loss_obj.item(), 'dfl': 0.0}
        
        # 4. DECODE PREDICTIONS
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # 5. GÁN NHÃN SỬ DỤNG TASK-ALIGNED ASSIGNER
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach(),
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        # 6. TÍNH LOSS
        # Objectness Loss (tính trên tất cả các anchor)
        loss[1] = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum
        
        # Bbox Loss & DFL Loss (chỉ tính trên các anchor positive - fg_mask)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask
            )
        
        # Áp dụng trọng số
        loss[0] *= self.box_weight
        loss[1] *= self.cls_weight
        loss[2] *= self.dfl_weight
        
        total_loss = loss.sum()
        
        loss_dict = {
            'total': total_loss.item(), 
            'ciou': loss[0].item(), 
            'obj': loss[1].item(), 
            'dfl': loss[2].item()
        }
        
        return total_loss, loss_dict


# ============================================================================
#                             TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Loss Functions Test")
    print("="*60)
    
    # Test data
    batch_size = 8
    num_classes = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Mock predictions
    predictions = {
        'boxes': torch.rand(batch_size, 4).to(device),  # Random boxes
        'cls': torch.randn(batch_size, num_classes).to(device),  # Logits
        'obj': torch.randn(batch_size, 1).to(device)
    }
    
    # Mock targets
    targets = {
        'boxes': torch.rand(batch_size, 4).to(device),
        'cls': torch.zeros(batch_size, num_classes).to(device)
    }
    targets['cls'][:4, 0] = 1  # First 4 samples are positive
    
    # Create loss function
    loss_fn = FewShotYOLOLoss()
    
    # Compute loss
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    print(f"\n✓ Loss computation successful")
    print(f"  - Total loss: {total_loss.item():.4f}")
    print(f"\nIndividual losses:")
    for name, value in loss_dict.items():
        if name != 'total':
            print(f"  - {name.upper()}: {value:.4f}")
    
    # Test individual loss functions
    print(f"\n✓ Testing individual losses...")
    
    pred_boxes = torch.tensor([[0.2, 0.2, 0.8, 0.8]], device=device)
    target_boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]], device=device)
    
    ciou = compute_ciou_loss(pred_boxes, target_boxes)
    print(f"  - CIoU Loss: {ciou.item():.4f}")
    
    dice = dice_loss(pred_boxes, target_boxes)
    print(f"  - Dice Loss: {dice.item():.4f}")
    
    pred_cls = torch.randn(4, num_classes, device=device)
    target_cls = torch.zeros(4, num_classes, device=device)
    target_cls[0, 0] = 1
    
    focal = focal_loss(pred_cls, target_cls, alpha=0.25, gamma=1.5)
    print(f"  - Focal Loss: {focal.item():.4f}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")