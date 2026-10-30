
from pathlib import Path

import torch
from ultralytics import YOLO


# --------------------
# 1. Configuration
# --------------------
ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = (
    ROOT / "outputs" / "yolo" / "training"
    / "yolov8n_coco128" / "weights" / "best.pt"
)

# 可以替换成其他图片
SOURCE = ROOT / "src" / "bus.jpg"

OUTPUT_DIR = ROOT / "outputs" / "yolo" / "predictions"

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25


# --------------------
# 2. Prediction
# --------------------
def main():

    # Check model weights
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Model weights not found: {WEIGHTS}"
        )

    # Check input image
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Input image not found: {SOURCE}"
        )

    # Select device
    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print("Device: CUDA")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Device: CPU")

    # Load trained YOLO model
    model = YOLO(str(WEIGHTS))

    print("Model loaded:", WEIGHTS.name)
    print("Input image:", SOURCE.name)

    # Run inference
    results = model.predict(
        source=str(SOURCE),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        device=device,
        save=True,
        project=str(OUTPUT_DIR),
        name="finetuned_demo",
        exist_ok=True,
        verbose=True,
    )

    # Display detection results
    for result in results:

        print("\nDetection Results:")

        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected.")
            continue

        for box in result.boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().tolist()
            )

            class_name = result.names[class_id]

            print(
                f"Class: {class_name:<12} "
                f"Confidence: {confidence:.3f} "
                f"BBox: "
                f"[{x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f}]"
            )

    print("\nPrediction completed.")
    print("Results saved to:", OUTPUT_DIR / "finetuned_demo")


if __name__ == "__main__":
    main()
