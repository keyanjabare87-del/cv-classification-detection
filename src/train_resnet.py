from torchvision.models import resnet18
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from tqdm import tqdm


# --------------------
# 1. 配置
# --------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
VAL_SIZE = 5000

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

for folder in [CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------
# 2. 数据集
# --------------------
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_aug_dataset = datasets.CIFAR10(
    DATA_DIR, train=True, download=True,
    transform=train_transform
)
train_eval_dataset = datasets.CIFAR10(
    DATA_DIR, train=True, download=True,
    transform=eval_transform
)
test_dataset = datasets.CIFAR10(
    DATA_DIR, train=False, download=True,
    transform=eval_transform
)

# 固定划分，确保实验可复现
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(
    len(train_aug_dataset), generator=generator
).tolist()

val_indices = indices[:VAL_SIZE]
train_indices = indices[VAL_SIZE:]

train_dataset = Subset(train_aug_dataset, train_indices)
val_dataset = Subset(train_eval_dataset, val_indices)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0,
    pin_memory=(device.type == "cuda")
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

classes = train_aug_dataset.classes
print("Classes:", classes)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))



from torchvision.models import resnet18


# --------------------
# 3. ResNet-18 模型
# --------------------
def build_resnet18(num_classes=10):
    model = resnet18(weights=None)

    # CIFAR-10 图片是 32×32。
    # 使用 3×3 卷积和步长 1，保留更多初始空间信息。
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # 去掉原本为大尺寸图片设计的 7×7 最大池化
    model.maxpool = nn.Identity()

    # 将 ImageNet 的 1000 类输出改成 CIFAR-10 的 10 类
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


model = build_resnet18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)



# --------------------
# 4. 训练和验证
# --------------------
def run_epoch(loader, training):
    model.train(training)
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(
        all_labels, all_preds, average="macro"
    )

    return avg_loss, acc, macro_f1


history = []
best_val_acc = -1.0
best_path = CHECKPOINT_DIR / "resnet18_best.pth"

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc, train_f1 = run_epoch(
        train_loader, training=True
    )
    val_loss, val_acc, val_f1 = run_epoch(
        val_loader, training=False
    )

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train loss={train_loss:.4f} "
        f"acc={train_acc:.4f} | "
        f"val loss={val_loss:.4f} "
        f"acc={val_acc:.4f} "
        f"macro-F1={val_f1:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "classes": classes,
            "seed": SEED,
        }, best_path)
        print("Saved best checkpoint:", best_path)


# 保存训练日志
log_path = LOG_DIR / "resnet18_history.csv"
with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)


# --------------------
# 5. 载入最佳模型并测试
# --------------------
checkpoint = torch.load(
    best_path, map_location=device, weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(
    all_labels, all_preds, average="macro"
)

print("\nBest epoch:", checkpoint["epoch"])
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro-F1: {test_f1:.4f}")
print("\nClassification report:")
print(classification_report(
    all_labels, all_preds, target_names=classes,
    digits=4, zero_division=0
))

cm = confusion_matrix(
    all_labels, all_preds, labels=list(range(10))
)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm)
fig.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(10),
    yticks=np.arange(10),
    xticklabels=classes,
    yticklabels=classes,
    xlabel="Predicted label",
    ylabel="True label",
    title="resnet18 Confusion Matrix",
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "resnet18_confusion_matrix.png", dpi=160)
plt.close(fig)

epochs = [r["epoch"] for r in history]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train")
axes[0].plot(epochs, [r["val_loss"] for r in history], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs, [r["train_acc"] for r in history], label="Train")
axes[1].plot(epochs, [r["val_acc"] for r in history], label="Validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

fig.tight_layout()
fig.savefig(FIGURE_DIR / "resnet18_training_curves.png", dpi=160)
plt.close(fig)

print("\nArtifacts saved:")
print("Checkpoint:", best_path)
print("History:", log_path)
print("Confusion matrix:", FIGURE_DIR / "resnet18_confusion_matrix.png")
print("Training curves:", FIGURE_DIR / "cresnet18_training_curves.png")
