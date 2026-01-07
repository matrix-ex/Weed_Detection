from ultralytics import YOLO
import os

def train_model():
    """
    Train YOLOv8 model on weed detection dataset
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load a pretrained YOLOv8 model (nano version for faster training)
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='data/data.yaml',      # path to dataset YAML
        epochs=20,                  # number of epochs
        imgsz=640,                   # image size
        batch=16,                    # batch size
        name='weed_detection',       # project name
        patience=50,                 # early stopping patience
        save=True,                   # save checkpoints
        device=0,                    # use GPU 0 (or 'cpu' for CPU training)
        workers=8,                   # number of worker threads
        pretrained=True,             # use pretrained weights
        optimizer='Adam',            # optimizer
        verbose=True,                # verbose output
        seed=42,                     # random seed
        deterministic=True,          # deterministic training
        project='runs/train',        # save results to project/name
        exist_ok=True,               # overwrite existing project
        cos_lr=True,                 # cosine learning rate scheduler
        lr0=0.01,                    # initial learning rate
        lrf=0.01,                    # final learning rate
        momentum=0.937,              # SGD momentum
        weight_decay=0.0005,         # optimizer weight decay
        warmup_epochs=3.0,           # warmup epochs
        warmup_momentum=0.8,         # warmup initial momentum
        box=7.5,                     # box loss gain
        cls=0.5,                     # cls loss gain
        val=True,                    # validate during training
    )
    
    # Save the best model to models directory
    best_model_path = 'runs/train/weed_detection/weights/best.pt'
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, 'models/best.pt')
        print(f"\n✓ Best model saved to: models/best.pt")
    
    # Print training results
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Results saved to: runs/train/weed_detection")
    print(f"Best model: models/best.pt")
    
    return results

def validate_model():
    """
    Validate the trained model on test set
    """
    if not os.path.exists('models/best.pt'):
        print("Error: Trained model not found. Please train the model first.")
        return
    
    # Load the trained model
    model = YOLO('models/best.pt')
    
    # Validate
    metrics = model.val(data='data/data.yaml')
    
    print("\n" + "="*50)
    print("Validation Metrics:")
    print("="*50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train weed detection model')
    parser.add_argument('--validate', action='store_true', help='Validate model after training')
    args = parser.parse_args()
    
    # Train the model
    print("Starting model training...")
    train_model()
    
    # Optionally validate
    if args.validate:
        print("\nValidating model...")
        validate_model()