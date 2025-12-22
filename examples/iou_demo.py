"""
Simple IoU Demonstration (No TensorFlow Required)

This demonstrates the core concept of Intersection over Union
which is fundamental to YOLO object detection.
"""


def iou_simple(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    
    Args:
        box1: tuple (x1, y1, x2, y2) - coordinates of box 1
        box2: tuple (x1, y1, x2, y2) - coordinates of box 2
        
    Returns:
        float: IoU value between 0 and 1
    """
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1_box1, x1_box2)
    yi1 = max(y1_box1, y1_box2)
    xi2 = min(x2_box1, x2_box2)
    yi2 = min(y2_box1, y2_box2)
    
    # Calculate intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def visualize_boxes(box1, box2):
    """Create ASCII visualization of two boxes"""
    x1_b1, y1_b1, x2_b1, y2_b1 = box1
    x1_b2, y1_b2, x2_b2, y2_b2 = box2
    
    # Find bounding region
    min_x = min(x1_b1, x1_b2)
    max_x = max(x2_b1, x2_b2)
    min_y = min(y1_b1, y1_b2)
    max_y = max(y2_b1, y2_b2)
    
    width = int(max_x - min_x) + 2
    height = int(max_y - min_y) + 2
    
    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Draw box 1 with 'A'
    for y in range(int(y1_b1 - min_y), int(y2_b1 - min_y) + 1):
        for x in range(int(x1_b1 - min_x), int(x2_b1 - min_x) + 1):
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = 'A'
    
    # Draw box 2 with 'B' (overlapping areas become 'X')
    for y in range(int(y1_b2 - min_y), int(y2_b2 - min_y) + 1):
        for x in range(int(x1_b2 - min_x), int(x2_b2 - min_x) + 1):
            if 0 <= y < height and 0 <= x < width:
                if grid[y][x] == 'A':
                    grid[y][x] = 'X'  # Overlap
                else:
                    grid[y][x] = 'B'
    
    # Print grid
    for row in grid:
        print(''.join(row))


def demo_scenarios():
    """Demonstrate different IoU scenarios"""
    
    print("=" * 70)
    print("YOLO Object Detection - Intersection over Union (IoU) Demo")
    print("=" * 70)
    print("\nIoU is a key metric in object detection that measures how much")
    print("two bounding boxes overlap. It ranges from 0 (no overlap) to 1")
    print("(perfect overlap).")
    print()
    
    scenarios = [
        {
            "name": "Scenario 1: High Overlap (Same Object)",
            "box1": (10, 10, 50, 50),
            "box2": (15, 15, 55, 55),
            "description": "Two boxes detecting the same object with high overlap"
        },
        {
            "name": "Scenario 2: Moderate Overlap",
            "box1": (10, 10, 40, 40),
            "box2": (30, 30, 60, 60),
            "description": "Boxes with moderate overlap"
        },
        {
            "name": "Scenario 3: Low Overlap",
            "box1": (10, 10, 30, 30),
            "box2": (25, 25, 50, 50),
            "description": "Boxes with low overlap"
        },
        {
            "name": "Scenario 4: No Overlap (Different Objects)",
            "box1": (10, 10, 30, 30),
            "box2": (50, 50, 70, 70),
            "description": "Two boxes detecting different objects"
        },
        {
            "name": "Scenario 5: One Box Inside Another",
            "box1": (10, 10, 100, 100),
            "box2": (30, 30, 70, 70),
            "description": "Smaller box completely inside larger box"
        }
    ]
    
    for scenario in scenarios:
        print("\n" + "-" * 70)
        print(f"\n{scenario['name']}")
        print(scenario['description'])
        print(f"\nBox 1 (A): {scenario['box1']}")
        print(f"Box 2 (B): {scenario['box2']}")
        
        iou_value = iou_simple(scenario['box1'], scenario['box2'])
        
        print(f"\nIoU: {iou_value:.4f}")
        
        # Interpretation
        if iou_value > 0.7:
            status = "🟢 HIGH - Likely same object (NMS would remove one)"
        elif iou_value > 0.3:
            status = "🟡 MODERATE - Possible overlap"
        elif iou_value > 0:
            status = "🟠 LOW - Different objects or slight overlap"
        else:
            status = "⚪ NONE - Completely different objects"
        
        print(f"Status: {status}")
        
        print("\nVisualization (A=Box1, B=Box2, X=Overlap):")
        try:
            visualize_boxes(scenario['box1'], scenario['box2'])
        except:
            print("  [Visualization skipped for this configuration]")
    
    print("\n" + "=" * 70)
    print("\nKey Takeaways:")
    print("=" * 70)
    print("• IoU > 0.5: Typically used as NMS threshold to remove duplicate detections")
    print("• IoU = 1.0: Perfect match (boxes are identical)")
    print("• IoU = 0.0: No overlap (completely different objects)")
    print("• NMS keeps the box with highest confidence and removes others with IoU > threshold")
    print()


def interactive_demo():
    """Allow user to input their own boxes"""
    print("\n" + "=" * 70)
    print("Interactive IoU Calculator")
    print("=" * 70)
    print("\nTry your own boxes! (Press Enter to skip)")
    print("Format: x1 y1 x2 y2 (e.g., 10 10 50 50)")
    
    try:
        box1_input = input("\nBox 1 coordinates: ").strip()
        if box1_input:
            box1 = tuple(map(int, box1_input.split()))
            
            box2_input = input("Box 2 coordinates: ").strip()
            if box2_input:
                box2 = tuple(map(int, box2_input.split()))
                
                iou_value = iou_simple(box1, box2)
                
                print(f"\nBox 1: {box1}")
                print(f"Box 2: {box2}")
                print(f"IoU: {iou_value:.4f}")
                
                print("\nVisualization:")
                visualize_boxes(box1, box2)
    except:
        print("Skipping interactive mode...")


if __name__ == "__main__":
    demo_scenarios()
    interactive_demo()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nThis concept is central to YOLO's Non-Max Suppression (NMS),")
    print("which eliminates redundant bounding boxes for the same object.")
    print()
