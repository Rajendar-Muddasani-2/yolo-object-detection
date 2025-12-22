# YOLO Object Detection

A simplified implementation of the **YOLO (You Only Look Once)** object detection algorithm in Python using TensorFlow. This project demonstrates the core concepts of YOLO including box filtering, Intersection over Union (IoU), and Non-Max Suppression (NMS).

## 🎯 About YOLO

YOLO is a state-of-the-art, real-time object detection system that can detect multiple objects in an image in a single forward pass through a neural network. Unlike traditional methods that apply a classifier to different regions of an image, YOLO looks at the entire image only once (hence "You Only Look Once") and predicts bounding boxes and class probabilities directly.

### Key Features

- **Real-time Detection**: YOLO achieves high accuracy while maintaining real-time performance
- **Single Neural Network**: The entire detection pipeline is a single network, making it easy to optimize
- **Global Context**: YOLO sees the entire image during training and test time, encoding contextual information

## 📚 References

This implementation is based on the original YOLO papers:
- [Redmon et al., 2016 - You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [Redmon and Farhadi, 2016 - YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.7
tensorflow >= 2.0
numpy
pillow
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/rajendarmuddasani/yolo-object-detection.git
cd yolo-object-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Examples

Run the demonstration script to see YOLO components in action:

```bash
python examples/simple_detection.py
```

Run unit tests:

```bash
python tests/test_yolo_utils.py
```

## 📖 How It Works

### 1. Box Filtering

YOLO predicts a large number of bounding boxes (19×19×5 = 1,805 boxes for a 19×19 grid with 5 anchor boxes). Most of these boxes have low confidence scores. Box filtering removes boxes with confidence below a threshold.

```python
from yolo_utils import yolo_filter_boxes

scores, boxes, classes = yolo_filter_boxes(
    boxes, box_confidence, box_class_probs, threshold=0.6
)
```

### 2. Intersection over Union (IoU)

IoU measures the overlap between two bounding boxes. It's calculated as:

```
IoU = (Area of Overlap) / (Area of Union)
```

```python
from yolo_utils import iou

iou_value = iou(box1, box2)
```

### 3. Non-Max Suppression (NMS)

Even after filtering, multiple boxes may detect the same object. NMS removes redundant boxes by:
1. Selecting the box with the highest score
2. Removing all other boxes with high overlap (IoU > threshold)
3. Repeating until no boxes remain

```python
from yolo_utils import yolo_non_max_suppression

scores, boxes, classes = yolo_non_max_suppression(
    scores, boxes, classes, max_boxes=10, iou_threshold=0.5
)
```

## 🏗️ Project Structure

```
yolo-object-detection/
├── src/
│   └── yolo_utils.py          # Core YOLO utilities
├── examples/
│   └── simple_detection.py    # Demonstration scripts
├── tests/
│   └── test_yolo_utils.py     # Unit tests
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## 🔍 Core Components

### `yolo_filter_boxes`
Filters boxes by thresholding on class confidence scores.

**Input:**
- `boxes`: Tensor of shape (19, 19, 5, 4)
- `box_confidence`: Tensor of shape (19, 19, 5, 1)
- `box_class_probs`: Tensor of shape (19, 19, 5, 80)
- `threshold`: Confidence threshold (default: 0.6)

**Output:**
- `scores`: Filtered confidence scores
- `boxes`: Filtered box coordinates
- `classes`: Filtered class predictions

### `iou`
Computes Intersection over Union between two boxes.

**Input:**
- `box1`: Tuple (x1, y1, x2, y2)
- `box2`: Tuple (x1, y1, x2, y2)

**Output:**
- `iou`: Float value between 0 and 1

### `yolo_non_max_suppression`
Applies Non-Max Suppression to remove redundant boxes.

**Input:**
- `scores`: Confidence scores
- `boxes`: Box coordinates
- `classes`: Class predictions
- `max_boxes`: Maximum number of boxes to keep (default: 10)
- `iou_threshold`: IoU threshold for suppression (default: 0.5)

**Output:**
- `scores`: NMS-filtered scores
- `boxes`: NMS-filtered boxes
- `classes`: NMS-filtered classes

## 📊 Example Output

```
Demo 1: Filtering Boxes by Confidence Threshold
================================================================
Original boxes: 19 x 19 x 5 = 1805 boxes
After filtering (threshold=0.5): 1789 boxes

Demo 2: Intersection over Union (IoU)
================================================================
Test 1 - Intersecting boxes:
  Box 1: (2, 1, 4, 3)
  Box 2: (1, 2, 3, 4)
  IoU: 0.1429

Demo 3: Non-Max Suppression (NMS)
================================================================
Before NMS: 8 boxes
After NMS (IoU threshold=0.5): 4 boxes
NMS removed 4 overlapping boxes
```

## 🧪 Testing

The project includes comprehensive unit tests:

```bash
cd tests
python test_yolo_utils.py
```

Tests cover:
- Box filtering functionality
- IoU calculation accuracy
- Non-Max Suppression behavior
- Tensor type validation

## 🎓 Learning Objectives

This project helps you understand:

1. **Object Detection Fundamentals**
   - How YOLO predicts bounding boxes
   - Grid-based detection approach
   - Anchor boxes concept

2. **Box Filtering Techniques**
   - Confidence thresholding
   - Class probability scoring

3. **Intersection over Union (IoU)**
   - Geometric intersection calculation
   - Evaluating box overlap

4. **Non-Max Suppression**
   - Removing duplicate detections
   - Selecting best predictions

5. **TensorFlow Operations**
   - Broadcasting
   - Boolean masking
   - Tensor manipulation

## 🛠️ Use Cases

This implementation can be extended for various applications:

- **Autonomous Driving**: Vehicle and pedestrian detection
- **Surveillance**: Real-time monitoring and threat detection
- **Retail**: Product recognition and inventory management
- **Healthcare**: Medical image analysis
- **Agriculture**: Crop and pest detection

## 📈 Performance Considerations

- **Grid Size**: Larger grids (e.g., 19×19) provide finer spatial resolution
- **Anchor Boxes**: More anchors can detect more objects but increase computation
- **Threshold Values**: 
  - Lower confidence threshold → more detections but more false positives
  - Higher IoU threshold → fewer boxes removed by NMS

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original YOLO papers by Joseph Redmon et al.
- TensorFlow team for the excellent deep learning framework
- Deep Learning community for continuous improvements to object detection

## 📧 Contact

Rajendarムドサニ - [@rajendarmuddasani](https://github.com/rajendarmuddasani)

Project Link: [https://github.com/rajendarmuddasani/yolo-object-detection](https://github.com/rajendarmuddasani/yolo-object-detection)

---

⭐ **Star this repository if you find it helpful!**
