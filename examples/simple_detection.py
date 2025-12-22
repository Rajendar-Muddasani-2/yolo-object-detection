"""
Simple YOLO Object Detection Example

This script demonstrates how to use the YOLO utilities for object detection.
It includes examples of:
1. Filtering boxes by confidence threshold
2. Computing Intersection over Union (IoU)
3. Applying Non-Max Suppression
"""

import sys
sys.path.append('../src')

import numpy as np
import tensorflow as tf
from yolo_utils import (
    yolo_filter_boxes,
    iou,
    yolo_non_max_suppression
)


def demo_filter_boxes():
    """Demonstrate box filtering by confidence threshold"""
    print("=" * 60)
    print("Demo 1: Filtering Boxes by Confidence Threshold")
    print("=" * 60)
    
    # Generate random test data
    tf.random.set_seed(42)
    box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
    boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
    box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
    
    # Apply filtering
    scores, filtered_boxes, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=0.5
    )
    
    print(f"\nOriginal boxes: 19 x 19 x 5 = {19*19*5} boxes")
    print(f"After filtering (threshold=0.5): {len(scores)} boxes")
    print(f"\nTop 3 scores: {scores[:3].numpy()}")
    print(f"Top 3 classes: {classes[:3].numpy()}")
    print(f"Scores shape: {scores.shape}")
    print(f"Boxes shape: {filtered_boxes.shape}")
    print(f"Classes shape: {classes.shape}")


def demo_iou():
    """Demonstrate Intersection over Union calculation"""
    print("\n" + "=" * 60)
    print("Demo 2: Intersection over Union (IoU)")
    print("=" * 60)
    
    # Test case 1: Intersecting boxes
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)
    iou_value = iou(box1, box2)
    print(f"\nTest 1 - Intersecting boxes:")
    print(f"  Box 1: {box1}")
    print(f"  Box 2: {box2}")
    print(f"  IoU: {iou_value:.4f}")
    
    # Test case 2: Non-intersecting boxes
    box1 = (1, 2, 3, 4)
    box2 = (5, 6, 7, 8)
    iou_value = iou(box1, box2)
    print(f"\nTest 2 - Non-intersecting boxes:")
    print(f"  Box 1: {box1}")
    print(f"  Box 2: {box2}")
    print(f"  IoU: {iou_value:.4f}")
    
    # Test case 3: Partially overlapping boxes
    box1 = (0, 0, 4, 4)
    box2 = (2, 2, 6, 6)
    iou_value = iou(box1, box2)
    print(f"\nTest 3 - Partially overlapping boxes:")
    print(f"  Box 1: {box1}")
    print(f"  Box 2: {box2}")
    print(f"  IoU: {iou_value:.4f}")
    
    # Test case 4: One box inside another
    box1 = (0, 0, 10, 10)
    box2 = (3, 3, 7, 7)
    iou_value = iou(box1, box2)
    print(f"\nTest 4 - One box inside another:")
    print(f"  Box 1: {box1}")
    print(f"  Box 2: {box2}")
    print(f"  IoU: {iou_value:.4f}")


def demo_nms():
    """Demonstrate Non-Max Suppression"""
    print("\n" + "=" * 60)
    print("Demo 3: Non-Max Suppression (NMS)")
    print("=" * 60)
    
    # Create sample boxes with overlaps
    scores = tf.constant([0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.55, 0.5], dtype=tf.float32)
    boxes = tf.constant([
        [10, 10, 50, 50],   # Box 1 (high score)
        [12, 12, 52, 52],   # Box 2 (overlaps with 1)
        [15, 15, 55, 55],   # Box 3 (overlaps with 1,2)
        [100, 100, 150, 150],  # Box 4 (different location)
        [102, 102, 152, 152],  # Box 5 (overlaps with 4)
        [200, 200, 250, 250],  # Box 6 (different location)
        [10, 100, 50, 150],    # Box 7 (different location)
        [12, 102, 52, 152]     # Box 8 (overlaps with 7)
    ], dtype=tf.float32)
    classes = tf.constant([0, 0, 0, 1, 1, 2, 3, 3], dtype=tf.int64)
    
    print(f"\nBefore NMS: {len(scores)} boxes")
    print(f"Scores: {scores.numpy()}")
    
    # Apply NMS
    nms_scores, nms_boxes, nms_classes = yolo_non_max_suppression(
        scores, boxes, classes, max_boxes=10, iou_threshold=0.5
    )
    
    print(f"\nAfter NMS (IoU threshold=0.5): {len(nms_scores)} boxes")
    print(f"Remaining scores: {nms_scores.numpy()}")
    print(f"Remaining classes: {nms_classes.numpy()}")
    print(f"\nNMS removed {len(scores) - len(nms_scores)} overlapping boxes")


def demo_complete_pipeline():
    """Demonstrate complete YOLO pipeline"""
    print("\n" + "=" * 60)
    print("Demo 4: Complete YOLO Detection Pipeline")
    print("=" * 60)
    
    # Simulate YOLO output
    tf.random.set_seed(123)
    grid_h, grid_w, n_anchors, n_classes = 19, 19, 5, 80
    
    # Generate random predictions
    box_confidence = tf.nn.sigmoid(tf.random.normal([grid_h, grid_w, n_anchors, 1]))
    boxes = tf.random.normal([grid_h, grid_w, n_anchors, 4], mean=100, stddev=50)
    box_class_probs = tf.nn.softmax(tf.random.normal([grid_h, grid_w, n_anchors, n_classes]))
    
    print(f"\nInput shape:")
    print(f"  Grid: {grid_h}x{grid_w}")
    print(f"  Anchor boxes: {n_anchors}")
    print(f"  Total predictions: {grid_h * grid_w * n_anchors}")
    
    # Step 1: Filter by threshold
    scores, filtered_boxes, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=0.6
    )
    print(f"\nStep 1 - After confidence filtering: {len(scores)} boxes")
    
    # Step 2: Apply NMS
    final_scores, final_boxes, final_classes = yolo_non_max_suppression(
        scores, filtered_boxes, classes, max_boxes=10, iou_threshold=0.5
    )
    print(f"Step 2 - After NMS: {len(final_scores)} boxes")
    
    print(f"\nFinal detections:")
    for i in range(min(5, len(final_scores))):
        print(f"  Detection {i+1}:")
        print(f"    Class: {final_classes[i].numpy()}")
        print(f"    Score: {final_scores[i].numpy():.4f}")
        print(f"    Box: {final_boxes[i].numpy()}")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("YOLO Object Detection - Demonstration Suite")
    print("=" * 60)
    print("\nThis demo showcases the key components of YOLO object detection:")
    print("1. Box filtering by confidence threshold")
    print("2. Intersection over Union (IoU) calculation")
    print("3. Non-Max Suppression (NMS)")
    print("4. Complete detection pipeline")
    
    demo_filter_boxes()
    demo_iou()
    demo_nms()
    demo_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
