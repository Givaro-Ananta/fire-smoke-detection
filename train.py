"""
Fire & Smoke YOLOv8 Training Script
"""

import os
from ultralytics import YOLO


def train():
    # Configuration
    DATA_YAML = os.path.join(os.path.dirname(__file__), "data.yaml")
    MODEL = "yolov8n.pt"  # YOLOv8 Nano (fast, good for starting)
    EPOCHS = 1      # Quick test for CPU
    IMG_SIZE = 640
    BATCH_SIZE = 16
    PROJECT = os.path.join(os.path.dirname(__file__), "runs", "detect")
    NAME = "fire_smoke_detector"

    print("=" * 60)
    print("Fire & Smoke YOLOv8 Training")
    print("=" * 60)
    print(f"  Model:      {MODEL}")
    print(f"  Dataset:    {DATA_YAML}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print("=" * 60)

    # Load pretrained YOLOv8 model (Skipped as we are using the existing trained model)
    # model = YOLO(MODEL)

    # # Train
    # results = model.train(
    #     data=DATA_YAML,
    #     epochs=EPOCHS,
    #     imgsz=IMG_SIZE,
    #     batch=BATCH_SIZE,
    #     project=PROJECT,
    #     name=NAME,
    #     exist_ok=True,
    #     patience=10,       # Early stopping patience
    #     save=True,
    #     save_period=10,     # Save checkpoint every 10 epochs
    #     device="cpu",       # No CUDA GPU available, use CPU
    #     workers=2,
    #     verbose=True,
    #     plots=True,
    #     # Augmentation
    #     hsv_h=0.015,
    #     hsv_s=0.7,
    #     hsv_v=0.4,
    #     degrees=10.0,
    #     translate=0.1,
    #     scale=0.5,
    #     flipud=0.0,
    #     fliplr=0.5,
    #     mosaic=1.0,
    #     mixup=0.1,
    # )

    # Print results
    print("\n" + "=" * 60)
    print("Training Skipped! Using pre-trained model from runs folder.")
    print("=" * 60)

    best_model_path = os.path.join(PROJECT, NAME, "weights", "best.pt")
    print(f"  Best model: {best_model_path}")

    # Validate
    print("\nRunning validation...")
    model_best = YOLO(best_model_path)
    val_results = model_best.val(data=DATA_YAML, imgsz=IMG_SIZE)

    print(f"\n  mAP50:      {val_results.box.map50:.4f}")
    print(f"  mAP50-95:   {val_results.box.map:.4f}")
    print(f"  Precision:  {val_results.box.mp:.4f}")
    print(f"  Recall:     {val_results.box.mr:.4f}")
    print("=" * 60)

    return best_model_path


if __name__ == "__main__":
    train()
