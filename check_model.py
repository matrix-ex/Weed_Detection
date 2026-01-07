import os

print("Checking model status...")
print("="*50)

# Check if models directory exists
if os.path.exists('models'):
    print("✓ models/ directory exists")
    
    # List files in models directory
    files = os.listdir('models')
    if files:
        print(f"✓ Files in models/: {files}")
        
        # Check best.pt specifically
        if os.path.exists('models/best.pt'):
            size = os.path.getsize('models/best.pt') / (1024*1024)
            print(f"✓ best.pt exists ({size:.2f} MB)")
            
            # Try to load it
            try:
                from ultralytics import YOLO
                model = YOLO('models/best.pt')
                print("✓ Model loads successfully!")
                print(f"✓ Model classes: {model.names}")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
        else:
            print("✗ best.pt not found in models/")
    else:
        print("✗ models/ directory is empty")
else:
    print("✗ models/ directory doesn't exist")

# Check training results
print("\n" + "="*50)
print("Checking training results...")
weights_path = 'runs/train/weed_detection/weights'
if os.path.exists(weights_path):
    print(f"✓ Training weights found at: {weights_path}")
    weights = os.listdir(weights_path)
    print(f"Available weights: {weights}")
    
    # Copy best.pt if it's not in models/
    if 'best.pt' in weights and not os.path.exists('models/best.pt'):
        print("\nCopying best.pt to models/...")
        import shutil
        os.makedirs('models', exist_ok=True)
        shutil.copy(f'{weights_path}/best.pt', 'models/best.pt')
        print("✓ Model copied successfully!")
else:
    print(f"✗ No training weights found at: {weights_path}")