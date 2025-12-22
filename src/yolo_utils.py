"""
YOLO Object Detection Utilities
Based on "You Only Look Once" (YOLO) algorithm
References: 
- Redmon et al., 2016 (https://arxiv.org/abs/1506.02640)
- Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242)
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if highest class probability score < threshold,
                     then get rid of the corresponding box

    Returns:
        scores -- tensor containing the class probability score for selected boxes
        boxes -- tensor containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor containing the index of the class detected by the selected boxes
    """
    
    # Step 1: Compute box scores (element-wise product)
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes using the max box_scores
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)
    
    # Step 3: Create a filtering mask based on threshold
    filtering_mask = box_class_scores >= threshold
    
    # Step 4: Apply the mask to filter boxes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes


def iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
        box1 -- first box, tuple with coordinates (box1_x1, box1_y1, box1_x2, box1_y2)
        box2 -- second box, tuple with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
        
    Returns:
        iou -- intersection over union value
    """
    
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the intersection coordinates
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height
    
    # Calculate the Union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # Compute the IoU
    iou_value = inter_area / union_area if union_area > 0 else 0
    
    return iou_value


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
        scores -- tensor of shape (None,), output of yolo_filter_boxes()
        boxes -- tensor of shape (None, 4), output of yolo_filter_boxes()
        classes -- tensor of shape (None,), output of yolo_filter_boxes()
        max_boxes -- integer, maximum number of predicted boxes
        iou_threshold -- real value, "intersection over union" threshold for NMS filtering
    
    Returns:
        scores -- tensor of shape (, None), predicted score for each box
        boxes -- tensor of shape (4, None), predicted box coordinates
        classes -- tensor of shape (, None), predicted class for each box
    """
    
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    
    # Use tf.image.non_max_suppression() to get the list of indices
    nms_indices = tf.image.non_max_suppression(
        boxes, 
        scores, 
        max_boxes_tensor, 
        iou_threshold=iou_threshold
    )
    
    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), 
              max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    Converts the output of YOLO encoding (19x19x5x85) to predicted boxes with filtering
    
    Arguments:
        yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3))
        image_shape -- tensor of shape (2,) containing the input shape
        max_boxes -- integer, maximum number of predicted boxes
        score_threshold -- real value, threshold for score filtering
        iou_threshold -- real value, threshold for IoU filtering
    
    Returns:
        scores -- tensor of shape (None, ), predicted score for each box
        boxes -- tensor of shape (None, 4), predicted box coordinates
        classes -- tensor of shape (None,), predicted class for each box
    """
    
    # Retrieve outputs of the YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    
    # Convert boxes to corner coordinates
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    # Apply filtering
    scores, boxes, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold
    )
    
    # Scale boxes back to original image shape
    boxes = scale_boxes(boxes, image_shape)
    
    # Apply Non-max suppression
    scores, boxes, classes = yolo_non_max_suppression(
        scores, boxes, classes, max_boxes=max_boxes, iou_threshold=iou_threshold
    )
    
    return scores, boxes, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    """
    Convert YOLO box predictions to corner coordinates
    
    Arguments:
        box_xy -- tensor of shape (grid_height, grid_width, anchor_boxes, 2)
        box_wh -- tensor of shape (grid_height, grid_width, anchor_boxes, 2)
        
    Returns:
        boxes -- tensor of shape (grid_height, grid_width, anchor_boxes, 4)
    """
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    
    return tf.concat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ], axis=-1)


def scale_boxes(boxes, image_shape):
    """
    Scales the predicted boxes to match the original image shape
    
    Arguments:
        boxes -- tensor of shape (None, 4) containing box coordinates
        image_shape -- tensor of shape (2,) containing height and width
        
    Returns:
        scaled_boxes -- tensor of shape (None, 4) with scaled coordinates
    """
    height = image_shape[0]
    width = image_shape[1]
    image_dims = tf.stack([height, width, height, width])
    image_dims = tf.reshape(image_dims, [1, 4])
    
    boxes = boxes * image_dims
    
    return boxes


def preprocess_image(image_path, target_size=(608, 608)):
    """
    Preprocess image for YOLO model
    
    Arguments:
        image_path -- path to the image file
        target_size -- tuple, target size for the image
        
    Returns:
        image_data -- preprocessed image ready for model
    """
    from PIL import Image
    
    image = Image.open(image_path)
    image = image.resize(target_size, Image.BICUBIC)
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    
    return image_data


def draw_boxes(image, boxes, classes, scores, class_names):
    """
    Draw bounding boxes on image
    
    Arguments:
        image -- PIL Image object
        boxes -- numpy array of shape (None, 4) containing box coordinates
        classes -- numpy array of shape (None,) containing class indices
        scores -- numpy array of shape (None,) containing confidence scores
        class_names -- list of class names
        
    Returns:
        image -- PIL Image with boxes drawn
    """
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(image)
    
    # Define colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'cyan']
    
    for i, box in enumerate(boxes):
        y1, x1, y2, x2 = box
        class_id = int(classes[i])
        score = scores[i]
        
        color = colors[class_id % len(colors)]
        label = f'{class_names[class_id]}: {score:.2f}'
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        draw.text((x1, y1 - 10), label, fill=color)
    
    return image
