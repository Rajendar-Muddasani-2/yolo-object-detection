"""
Unit tests for YOLO utilities
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf
from yolo_utils import yolo_filter_boxes, iou, yolo_non_max_suppression


def test_yolo_filter_boxes():
    """Test box filtering functionality"""
    print("Testing yolo_filter_boxes...")
    
    tf.random.set_seed(10)
    box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
    boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
    box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
    
    scores, filtered_boxes, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=0.5
    )
    
    assert scores.shape[0] > 0, "Should have some boxes after filtering"
    assert filtered_boxes.shape[1] == 4, "Boxes should have 4 coordinates"
    assert len(scores) == len(classes), "Scores and classes should have same length"
    
    print("✓ yolo_filter_boxes test passed")
    return True


def test_iou():
    """Test IoU calculation"""
    print("Testing iou...")
    
    # Test intersecting boxes
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)
    iou_val = iou(box1, box2)
    assert 0 < iou_val < 1, "IoU should be between 0 and 1 for intersecting boxes"
    assert np.isclose(iou_val, 0.14285714, atol=1e-6), f"Expected ~0.1429, got {iou_val}"
    
    # Test non-intersecting boxes
    box1 = (1, 2, 3, 4)
    box2 = (5, 6, 7, 8)
    iou_val = iou(box1, box2)
    assert iou_val == 0, "IoU should be 0 for non-intersecting boxes"
    
    # Test boxes touching at vertices
    box1 = (1, 1, 2, 2)
    box2 = (2, 2, 3, 3)
    iou_val = iou(box1, box2)
    assert iou_val == 0, "IoU should be 0 for boxes touching at vertices"
    
    # Test boxes touching at edges
    box1 = (1, 1, 3, 3)
    box2 = (2, 3, 3, 4)
    iou_val = iou(box1, box2)
    assert iou_val == 0, "IoU should be 0 for boxes touching at edges"
    
    print("✓ iou test passed")
    return True


def test_yolo_non_max_suppression():
    """Test Non-Max Suppression"""
    print("Testing yolo_non_max_suppression...")
    
    # Create overlapping boxes
    scores = tf.constant([0.9, 0.85, 0.8, 0.6], dtype=tf.float32)
    boxes = tf.constant([
        [10, 10, 50, 50],
        [12, 12, 52, 52],  # Overlaps with first box
        [100, 100, 150, 150],  # Different location
        [102, 102, 152, 152]  # Overlaps with third box
    ], dtype=tf.float32)
    classes = tf.constant([0, 0, 1, 1], dtype=tf.int64)
    
    nms_scores, nms_boxes, nms_classes = yolo_non_max_suppression(
        scores, boxes, classes, max_boxes=10, iou_threshold=0.5
    )
    
    # Should remove overlapping boxes
    assert len(nms_scores) < len(scores), "NMS should remove some boxes"
    assert len(nms_scores) > 0, "Should keep at least some boxes"
    
    # Scores should be in descending order (implicitly from NMS)
    nms_scores_list = nms_scores.numpy().tolist()
    
    print(f"  Original boxes: {len(scores)}, After NMS: {len(nms_scores)}")
    print("✓ yolo_non_max_suppression test passed")
    return True


def test_tensor_types():
    """Test that functions return correct tensor types"""
    print("Testing tensor types...")
    
    tf.random.set_seed(42)
    box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4)
    boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4)
    box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4)
    
    scores, filtered_boxes, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=0.5
    )
    
    from tensorflow.python.framework.ops import EagerTensor
    assert isinstance(scores, EagerTensor), "scores should be EagerTensor"
    assert isinstance(filtered_boxes, EagerTensor), "boxes should be EagerTensor"
    assert isinstance(classes, EagerTensor), "classes should be EagerTensor"
    
    print("✓ tensor types test passed")
    return True


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "=" * 60)
    print("Running YOLO Unit Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_yolo_filter_boxes,
        test_iou,
        test_yolo_non_max_suppression,
        test_tensor_types
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
