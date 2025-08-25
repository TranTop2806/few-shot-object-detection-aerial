import torch
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from collections import defaultdict

class FSLEvaluator:
    """
    Comprehensive evaluator for Few-Shot Object Detection
    Includes mAP, precision, recall, and few-shot specific metrics
    """
    
    def __init__(self, gt_json_path, novel_classes):
        self.gt_json_path = gt_json_path
        self.novel_classes = novel_classes
        self.coco_gt = COCO(gt_json_path)
        
        # Get category mapping
        self.cat_name_to_id = {}
        for cat in self.coco_gt.dataset['categories']:
            if cat['name'] in novel_classes:
                self.cat_name_to_id[cat['name']] = cat['id']
    
    def evaluate_detections(self, predictions, confidence_threshold=0.5):
        """
        Evaluate detection results
        
        Args:
            predictions: List of dicts with keys: image_id, boxes, scores, labels
            confidence_threshold: Minimum confidence for positive detection
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        # Convert predictions to COCO format
        coco_results = []
        detection_count = defaultdict(int)
        
        for pred in predictions:
            image_id = pred['image_id']
            boxes = pred['boxes']  # [x1, y1, x2, y2]
            scores = pred['scores']
            labels = pred['labels']  # class names
            
            for box, score, label in zip(boxes, scores, labels):
                if score >= confidence_threshold and label in self.cat_name_to_id:
                    # Convert to COCO format [x, y, width, height]
                    x1, y1, x2, y2 = box
                    coco_box = [x1, y1, x2 - x1, y2 - y1]
                    
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': self.cat_name_to_id[label],
                        'bbox': coco_box,
                        'score': float(score)
                    })
                    detection_count[label] += 1
        
        if not coco_results:
            return {
                'mAP': 0.0,
                'mAP_50': 0.0,
                'mAP_75': 0.0,
                'per_class_AP': {cls: 0.0 for cls in self.novel_classes},
                'detection_count': dict(detection_count),
                'total_detections': 0
            }
        
        # Run COCO evaluation
        coco_dt = self.coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        
        # Filter to novel classes only
        novel_cat_ids = list(self.cat_name_to_id.values())
        coco_eval.params.catIds = novel_cat_ids
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            'mAP': float(coco_eval.stats[0]),  # mAP @ IoU=0.50:0.95
            'mAP_50': float(coco_eval.stats[1]),  # mAP @ IoU=0.50
            'mAP_75': float(coco_eval.stats[2]),  # mAP @ IoU=0.75
            'detection_count': dict(detection_count),
            'total_detections': sum(detection_count.values())
        }
        
        # Per-class AP
        per_class_AP = {}
        for i, cat_id in enumerate(novel_cat_ids):
            cat_name = [name for name, id in self.cat_name_to_id.items() if id == cat_id][0]
            if i < len(coco_eval.eval['precision']):
                # Average over IoU thresholds and area ranges
                precision = coco_eval.eval['precision'][i, :, :, 0, 2]
                per_class_AP[cat_name] = float(np.mean(precision[precision > -1]))
            else:
                per_class_AP[cat_name] = 0.0
        
        metrics['per_class_AP'] = per_class_AP
        
        return metrics
    
    def print_evaluation_summary(self, metrics):
        """Print formatted evaluation results"""
        print("\n" + "="*60)
        print("FSOD EVALUATION RESULTS")
        print("="*60)
        
        print(f"Overall Performance:")
        print(f"   mAP (IoU 0.5:0.95): {metrics['mAP']:.3f}")
        print(f"   mAP@50:            {metrics['mAP_50']:.3f}")
        print(f"   mAP@75:            {metrics['mAP_75']:.3f}")
        
        print(f"Per-Class Average Precision:")
        for class_name, ap in metrics['per_class_AP'].items():
            print(f"   {class_name:20}: {ap:.3f}")
        
        print(f"Detection Statistics:")
        print(f"   Total detections: {metrics['total_detections']}")
        for class_name, count in metrics['detection_count'].items():
            print(f"   {class_name:20}: {count} detections")
        
        print("="*60)

def run_comprehensive_evaluation():
    """Run comprehensive evaluation on test set"""
    
    print("Starting comprehensive FSOD evaluation...")
    
    test_json_path = "data/NWPU VHR-10/test/test.json"
    novel_classes = ['airplane', 'baseball-diamond', 'tennis-court']
    
    if not os.path.exists(test_json_path):
        print(f"Test annotations not found at {test_json_path}")
        return
    
    evaluator = FSLEvaluator(test_json_path, novel_classes)
        
    return evaluator

if __name__ == '__main__':
    run_comprehensive_evaluation()
