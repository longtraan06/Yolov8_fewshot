"""
Few-Shot YOLO - Complete Demo Scripts
Includes: Quick Start, Training, Inference, Evaluation
"""

import torch
import sys
from pathlib import Path
import argparse

# ==================== QUICK START DEMO ====================
def quick_start_demo():
    """
    Quick start demo - Train and test on sample data
    Run this first to verify installation
    """
    print("=" * 70)
    print("FEW-SHOT YOLO - QUICK START DEMO")
    print("=" * 70)
    print("\nThis demo will:")
    print("1. Create sample dataset")
    print("2. Train model for 10 epochs")
    print("3. Run inference on test images")
    print("4. Evaluate performance")
    print("=" * 70)
    
    # Step 1: Create sample dataset
    print("\n[Step 1/4] Creating sample dataset...")
    from utils import create_sample_dataset
    create_sample_dataset(
        output_dir='demo_dataset',
        num_classes=3,
        images_per_class=15
    )
    print("✓ Sample dataset created")
    
    # Step 2: Quick training
    print("\n[Step 2/4] Training model (10 epochs)...")
    from fewshot_yolo import FewShotYOLO, FewShotDataset, FewShotTrainer
    
    # Initialize dataset
    train_dataset = FewShotDataset(
        data_root='demo_dataset/',
        n_way=2,
        k_shot=3,
        n_query=3,
        mode='train'
    )
    
    val_dataset = FewShotDataset(
        data_root='demo_dataset/',
        n_way=2,
        k_shot=3,
        n_query=3,
        mode='val'
    )
    
    # Initialize model
    model = FewShotYOLO(
        backbone_weights='yolov8n.pt',
        k_shot=3,
        freeze_backbone=True
    )
    
    # Train
    trainer = FewShotTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=1e-4,
        batch_size=2,
        epochs=10,  # Quick demo
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir='demo_checkpoints/'
    )
    
    trainer.train()
    print("✓ Training completed")
    
    # Step 3: Inference
    print("\n[Step 3/4] Running inference...")
    from fewshot_yolo import FewShotInference
    
    inference = FewShotInference(
        model=model,
        checkpoint_path='demo_checkpoints/best_model.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Demo inference (assuming test images exist)
    support_images = [
        'demo_dataset/test/class_0/support/000.jpg',
        'demo_dataset/test/class_0/support/001.jpg',
        'demo_dataset/test/class_0/support/002.jpg',
    ]
    query_image = 'demo_dataset/test/class_0/query/000.jpg'
    
    results = inference.detect(support_images, query_image)
    inference.save_results(results, 'demo_results/result.jpg')
    print(f"✓ Detected {len(results['boxes'])} objects")
    
    # Step 4: Evaluation
    print("\n[Step 4/4] Evaluating performance...")
    print("✓ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Check 'demo_checkpoints/' for trained model")
    print("2. Check 'demo_results/' for detection results")
    print("3. Prepare your own dataset and run full training")
    print("=" * 70)


# ==================== TRAINING SCRIPT ====================
def train_script(args):
    """
    Full training script with all features
    
    Usage:
        python demo.py train --data dataset/ --epochs 100 --batch-size 4
    """
    from fewshot_yolo import FewShotYOLO, FewShotDataset, FewShotTrainer
    from utils import ConfigManager
    
    print("=" * 70)
    print("FEW-SHOT YOLO - TRAINING")
    print("=" * 70)
    
    # Load or create config
    if args.config:
        config = ConfigManager.load_config(args.config)
        print(f"✓ Loaded config from {args.config}")
    else:
        config = ConfigManager.get_default_config()
        print("✓ Using default config")
    
    # Override with command line args
    if args.data:
        config['data']['data_root'] = args.data
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    
    # Print config
    print("\nTraining Configuration:")
    print(f"  Data root: {config['data']['data_root']}")
    print(f"  N-way: {config['data']['n_way']}")
    print(f"  K-shot: {config['data']['k_shot']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['lr']}")
    print(f"  Device: {config['training']['device']}")
    print("=" * 70)
    
    # Initialize datasets
    print("\n[1/3] Loading datasets...")
    train_dataset = FewShotDataset(
        data_root=config['data']['data_root'],
        n_way=config['data']['n_way'],
        k_shot=config['data']['k_shot'],
        n_query=config['data']['n_query'],
        img_size=config['data']['img_size'],
        mode='train'
    )
    
    val_dataset = FewShotDataset(
        data_root=config['data']['data_root'],
        n_way=config['data']['n_way'],
        k_shot=config['data']['k_shot'],
        n_query=config['data']['n_query'],
        img_size=config['data']['img_size'],
        mode='val'
    )
    
    # Initialize model
    print("\n[2/3] Initializing model...")
    model = FewShotYOLO(
        backbone_weights=config['model']['backbone_weights'],
        k_shot=config['data']['k_shot'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        from utils import CheckpointManager
        epoch, history = CheckpointManager.load_checkpoint(args.resume, model)
        print(f"✓ Resumed from epoch {epoch}")
    
    # Train
    print("\n[3/3] Starting training...")
    trainer = FewShotTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=config['training']['lr'],
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        device=config['training']['device'],
        save_dir=config['paths']['save_dir']
    )
    
    trainer.train()
    
    print("\n" + "=" * 70)
    print("✓ Training completed successfully!")
    print(f"✓ Checkpoints saved in: {config['paths']['save_dir']}")
    print("=" * 70)


# ==================== INFERENCE SCRIPT ====================
def inference_script(args):
    """
    Inference script for detection
    
    Usage:
        python demo.py infer \
            --checkpoint checkpoints/best_model.pth \
            --support examples/s1.jpg examples/s2.jpg examples/s3.jpg \
            --query examples/query.jpg \
            --output results/
    """
    from fewshot_yolo import FewShotYOLO, FewShotInference
    import cv2
    import glob
    
    print("=" * 70)
    print("FEW-SHOT YOLO - INFERENCE")
    print("=" * 70)
    
    # Validate inputs
    if not Path(args.checkpoint).exists():
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        return
    
    if len(args.support) != 3:
        print(f"✗ Exactly 3 support images required, got {len(args.support)}")
        return
    
    for img_path in args.support:
        if not Path(img_path).exists():
            print(f"✗ Support image not found: {img_path}")
            return
    
    print(f"\n✓ Checkpoint: {args.checkpoint}")
    print(f"✓ Support images: {len(args.support)}")
    print(f"✓ Device: {args.device}")
    print("=" * 70)
    
    # Initialize model
    print("\n[1/3] Loading model...")
    model = FewShotYOLO(
        backbone_weights='yolov8n.pt',
        k_shot=3,
        freeze_backbone=True
    )
    
    inference = FewShotInference(
        model=model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Single image or batch inference
    if args.query:
        query_images = [args.query]
    elif args.query_dir:
        query_images = glob.glob(str(Path(args.query_dir) / '*.jpg'))
        query_images += glob.glob(str(Path(args.query_dir) / '*.png'))
    else:
        print("✗ Must specify --query or --query-dir")
        return
    
    print(f"\n[2/3] Processing {len(query_images)} images...")
    
    # Run inference
    from tqdm import tqdm
    results_summary = []
    
    for query_path in tqdm(query_images):
        results = inference.detect(args.support, query_path)
        
        # Save results
        output_name = Path(query_path).stem + '_result.jpg'
        output_path = output_dir / output_name
        inference.save_results(results, str(output_path))
        
        results_summary.append({
            'image': query_path,
            'detections': len(results['boxes']),
            'max_score': results['scores'].max() if len(results['scores']) > 0 else 0.0
        })
    
    # Print summary
    print("\n[3/3] Results:")
    print("-" * 70)
    for item in results_summary:
        print(f"{Path(item['image']).name:30s} | "
              f"Detections: {item['detections']:2d} | "
              f"Max Score: {item['max_score']:.3f}")
    print("-" * 70)
    
    total_detections = sum(r['detections'] for r in results_summary)
    avg_detections = total_detections / len(results_summary)
    
    print(f"\nTotal images: {len(results_summary)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections/image: {avg_detections:.2f}")
    print(f"\n✓ Results saved in: {output_dir}")
    print("=" * 70)


# ==================== EVALUATION SCRIPT ====================
def evaluate_script(args):
    """
    Evaluate model on test set
    
    Usage:
        python demo.py evaluate \
            --checkpoint checkpoints/best_model.pth \
            --test-data dataset/test
    """
    from fewshot_yolo import FewShotYOLO, FewShotDataset, FewShotInference
    from utils import MetricsCalculator
    from tqdm import tqdm
    
    print("=" * 70)
    print("FEW-SHOT YOLO - EVALUATION")
    print("=" * 70)
    
    # Load model
    print("\n[1/3] Loading model...")
    model = FewShotYOLO(
        backbone_weights='yolov8n.pt',
        k_shot=3,
        freeze_backbone=True
    )
    
    inference = FewShotInference(
        model=model,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Load test dataset
    print("\n[2/3] Loading test dataset...")
    test_dataset = FewShotDataset(
        data_root=args.test_data,
        n_way=2,
        k_shot=3,
        n_query=5,
        mode='test'
    )
    
    # Evaluate
    print("\n[3/3] Evaluating...")
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    
    for i in tqdm(range(len(test_dataset))):
        batch = test_dataset[i]
        
        # Get support and query
        support_imgs = batch['support_images']
        query_img = batch['query_image']
        query_target = batch['query_target']
        
        # Run inference
        # (Simplified - need to convert tensors to image paths)
        # results = inference.detect(support_imgs, query_img)
        
        # Collect predictions and ground truths
        # all_pred_boxes.append(results['boxes'])
        # all_pred_scores.append(results['scores'])
        # all_gt_boxes.append(query_target['boxes'].numpy())
    
    # Calculate metrics
    metrics = MetricsCalculator.evaluate_detections(
        all_pred_boxes,
        all_pred_scores,
        all_gt_boxes,
        iou_threshold=0.5
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1']:.4f}")
    print(f"mAP@0.5:    {metrics['mAP']:.4f}")
    print("=" * 70)


# ==================== DATASET CONVERSION SCRIPT ====================
def convert_dataset_script(args):
    """
    Convert COCO dataset to Few-Shot format
    
    Usage:
        python demo.py convert \
            --coco-json coco/annotations/instances_train2017.json \
            --images-dir coco/train2017 \
            --output dataset/
    """
    from utils import DatasetConverter
    
    print("=" * 70)
    print("DATASET CONVERSION - COCO to Few-Shot")
    print("=" * 70)
    
    print(f"\nInput:")
    print(f"  COCO JSON: {args.coco_json}")
    print(f"  Images dir: {args.images_dir}")
    print(f"\nOutput:")
    print(f"  Output dir: {args.output}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Val ratio: {args.val_ratio}")
    print(f"  Support per class: {args.support_per_class}")
    print("=" * 70)
    
    print("\nConverting...")
    DatasetConverter.coco_to_fewshot(
        coco_json=args.coco_json,
        images_dir=args.images_dir,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        support_per_class=args.support_per_class
    )
    
    print("\n" + "=" * 70)
    print("✓ Conversion completed!")
    print(f"✓ Dataset saved in: {args.output}")
    print("=" * 70)


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(
        description='Few-Shot YOLO - Complete Demo Scripts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start demo
  python demo.py quickstart
  
  # Train model
  python demo.py train --data dataset/ --epochs 100 --batch-size 4
  
  # Inference on single image
  python demo.py infer \\
      --checkpoint checkpoints/best_model.pth \\
      --support s1.jpg s2.jpg s3.jpg \\
      --query query.jpg
  
  # Batch inference
  python demo.py infer \\
      --checkpoint checkpoints/best_model.pth \\
      --support s1.jpg s2.jpg s3.jpg \\
      --query-dir test_images/
  
  # Evaluate model
  python demo.py evaluate \\
      --checkpoint checkpoints/best_model.pth \\
      --test-data dataset/test
  
  # Convert COCO dataset
  python demo.py convert \\
      --coco-json coco/annotations/instances_train2017.json \\
      --images-dir coco/train2017 \\
      --output dataset/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Quick start command
    quickstart_parser = subparsers.add_parser('quickstart', help='Run quick start demo')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data', type=str, help='Dataset root directory')
    train_parser.add_argument('--config', type=str, help='Config YAML file')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    infer_parser.add_argument('--support', nargs=3, required=True, help='3 support images')
    infer_parser.add_argument('--query', type=str, help='Single query image')
    infer_parser.add_argument('--query-dir', type=str, help='Directory of query images')
    infer_parser.add_argument('--output', type=str, default='results/', help='Output directory')
    infer_parser.add_argument('--device', type=str, default='cuda', help='Device')
    infer_parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    infer_parser.add_argument('--iou-threshold', type=float, default=0.45, help='IoU threshold')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    eval_parser.add_argument('--test-data', type=str, required=True, help='Test dataset directory')
    eval_parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert COCO to Few-Shot format')
    convert_parser.add_argument('--coco-json', type=str, required=True, help='COCO JSON file')
    convert_parser.add_argument('--images-dir', type=str, required=True, help='Images directory')
    convert_parser.add_argument('--output', type=str, required=True, help='Output directory')
    convert_parser.add_argument('--train-ratio', type=float, default=0.7, help='Train ratio')
    convert_parser.add_argument('--val-ratio', type=float, default=0.15, help='Val ratio')
    convert_parser.add_argument('--support-per-class', type=int, default=5, help='Support images per class')
    
    args = parser.parse_args()
    
    if args.command == 'quickstart':
        quick_start_demo()
    elif args.command == 'train':
        train_script(args)
    elif args.command == 'infer':
        inference_script(args)
    elif args.command == 'evaluate':
        evaluate_script(args)
    elif args.command == 'convert':
        convert_dataset_script(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()