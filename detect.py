from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

class WeedDetector:
    def __init__(self, model_path='models/best.pt'):
        """
        Initialize the weed detector
        
        Args:
            model_path: Path to trained YOLO model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        self.model = YOLO(model_path)
        print(f"Model loaded from: {model_path}")
    
    def detect_weeds(self, image_path, conf_threshold=0.25):
        """
        Detect weeds in an image and return coordinates
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            dict: Detection results with coordinates and annotated image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = image.shape[:2]
        
        # Run inference
        results = self.model(image, conf=conf_threshold)[0]
        
        # Extract detections
        detections = []
        annotated_image = image.copy()
        
        for box in results.boxes:
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Calculate center point (for laser targeting)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Get confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            # Store detection info
            detection = {
                'class': class_name,
                'confidence': confidence,
                'bbox': {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                },
                'center': {
                    'x': center_x,
                    'y': center_y,
                    'x_normalized': center_x / width,  # Normalized coordinates (0-1)
                    'y_normalized': center_y / height
                },
                'laser_coordinates': self.calculate_laser_coordinates(center_x, center_y, width, height)
            }
            detections.append(detection)
            
            # Draw bounding box
            # All detections are weeds since we only have 1 class
            color = (0, 255, 0)  # Green for weeds
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw center point (laser target)
            cv2.circle(annotated_image, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.circle(annotated_image, (center_x, center_y), 10, (255, 0, 0), 2)
            
            # Add label
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(annotated_image, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add coordinate text
            coord_text = f'({center_x}, {center_y})'
            cv2.putText(annotated_image, coord_text, (center_x - 40, center_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return {
            'detections': detections,
            'total_weeds': len(detections),  # All detections are weeds
            'annotated_image': annotated_image,
            'image_size': {'width': width, 'height': height}
        }
    
    def calculate_laser_coordinates(self, pixel_x, pixel_y, image_width, image_height):
        """
        Convert pixel coordinates to laser system coordinates
        This is a template - adjust based on your actual laser system
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            image_width, image_height: Image dimensions
            
        Returns:
            dict: Laser coordinates
        """
        # Example conversion - modify based on your laser system specs
        # Assuming laser system works in a coordinate system from 0-1000
        laser_x = int((pixel_x / image_width) * 1000)
        laser_y = int((pixel_y / image_height) * 1000)
        
        return {
            'x': laser_x,
            'y': laser_y,
            'unit': 'laser_units'  # Replace with your actual units
        }
    
    def save_annotated_image(self, annotated_image, output_path):
        """
        Save annotated image to file
        """
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to: {output_path}")

def main():
    """
    Test detection on a single image
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect weeds in image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='output.jpg', help='Output image path')
    args = parser.parse_args()
    
    # Initialize detector
    detector = WeedDetector(args.model)
    
    # Detect weeds
    print(f"Processing image: {args.image}")
    results = detector.detect_weeds(args.image, args.conf)
    
    # Print results
    print("\n" + "="*50)
    print("Detection Results:")
    print("="*50)
    print(f"Total weeds detected: {results['total_weeds']}")
    print(f"\nDetailed detections:")
    for i, det in enumerate(results['detections'], 1):
        print(f"\nWeed #{i}:")
        print(f"  Class: {det['class']}")
        print(f"  Confidence: {det['confidence']:.2%}")
        print(f"  Pixel Center: ({det['center']['x']}, {det['center']['y']})")
        print(f"  Laser Coords: ({det['laser_coordinates']['x']}, {det['laser_coordinates']['y']})")
    
    # Save annotated image
    detector.save_annotated_image(results['annotated_image'], args.output)

if __name__ == "__main__":
    main()