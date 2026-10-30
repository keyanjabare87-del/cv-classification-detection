
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "yolo" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        device = "cpu"

    # 加载 COCO 预训练的 YOLOv8n
    model = YOLO("yolov8n.pt")
    model.info()

    # 使用官方示例图片测试推理
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        imgsz=640,
        conf=0.25,
        device=device,
        save=True,
        project=str(OUTPUT_DIR),
        name="bus_demo",
        exist_ok=True,
        verbose=True,
    )

    # 输出检测类别、置信度和边界框
    for result in results:
        names = result.names
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print("No objects detected.")
            continue

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(
                f"class={names[class_id]}, "
                f"confidence={confidence:.3f}, "
                f"box={[round(v, 1) for v in (x1, y1, x2, y2)]}"
            )

    print("Prediction completed.")


if __name__ == "__main__":
    main()
