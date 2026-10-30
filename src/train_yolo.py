
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "outputs" / "yolo" / "training"

def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    # 从 COCO 预训练权重开始微调
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="coco128.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        project=str(PROJECT_DIR),
        name="yolov8n_coco128",
        exist_ok=True,
        plots=True,
        seed=42,
    )

    print("Training completed.")
    print("Results directory:", PROJECT_DIR / "yolov8n_coco128")
    print("Best weights:", PROJECT_DIR / "yolov8n_coco128" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
