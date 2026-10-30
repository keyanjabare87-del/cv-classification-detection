
import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from ultralytics import YOLO


# ====================================
# 1. Project configuration
# ====================================
ROOT = Path(__file__).resolve().parent

CLASSIFY_WEIGHTS = (
    ROOT / "outputs" / "checkpoints"
    / "resnet18_cosine_best.pth"
)

DETECT_WEIGHTS = (
    ROOT / "outputs" / "yolo" / "training"
    / "yolov8n_coco128" / "weights" / "best.pt"
)

DEFAULT_OUTPUT = ROOT / "outputs" / "predictions"

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ====================================
# 2. Build classification model
# ====================================
def build_resnet18(num_classes=10):
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def load_classification_model():
    if not CLASSIFY_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Classification weights not found: "
            f"{CLASSIFY_WEIGHTS}"
        )

    checkpoint = torch.load(
        CLASSIFY_WEIGHTS,
        map_location=DEVICE,
        weights_only=True
    )

    model = build_resnet18(num_classes=10)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# ====================================
# 3. Image classification
# ====================================
def classify_image(image_path, output_dir):
    model = load_classification_model()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    probabilities = probabilities.squeeze(0).cpu()

    top_probs, top_indices = torch.topk(
        probabilities, k=3
    )

    print("\n========== Classification Results ==========")
    print("Input:", image_path)
    print("Device:", DEVICE)

    for rank, (prob, index) in enumerate(
        zip(top_probs, top_indices), start=1
    ):
        class_name = CIFAR_CLASSES[index.item()]
        print(
            f"Top {rank}: {class_name:<12} "
            f"Probability: {prob.item():.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "classification_result.txt"

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Input: {image_path}\n")
        for rank, (prob, index) in enumerate(
            zip(top_probs, top_indices), start=1
        ):
            f.write(
                f"Top {rank}: "
                f"{CIFAR_CLASSES[index.item()]} "
                f"{prob.item():.4f}\n"
            )

    print("Result saved:", result_path)


# ====================================
# 4. Object detection
# ====================================
def detect_image(image_path, output_dir, conf):
    if not DETECT_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Detection weights not found: {DETECT_WEIGHTS}"
        )

    model = YOLO(str(DETECT_WEIGHTS))

    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=conf,
        device=0 if DEVICE.type == "cuda" else "cpu",
        save=True,
        project=str(output_dir),
        name="detection",
        exist_ok=True,
        verbose=False,
    )

    print("\n========== Detection Results ==========")
    print("Input:", image_path)
    print("Device:", DEVICE)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected.")
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = box.xyxy[0].cpu().tolist()

            print(
                f"Class: {result.names[class_id]:<15} "
                f"Confidence: {confidence:.4f} "
                f"BBox: {[round(x, 1) for x in bbox]}"
            )

    print("Detection visualization saved to:")
    print(output_dir / "detection")


# ====================================
# 5. Main entry
# ====================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified Computer Vision Predictor"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classify", "detect"],
        help="Prediction task"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output directory"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1")

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    print("Computer Vision Unified Predictor")
    print("----------------------------------")
    print("Task:", args.task)
    print("Image:", image_path)
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if args.task == "classify":
        classify_image(image_path, output_dir)

    elif args.task == "detect":
        detect_image(image_path, output_dir, args.conf)


if __name__ == "__main__":
    main()
