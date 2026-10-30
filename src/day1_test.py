import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 将图片转化为Tensor,并进行归一化
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True, # 打乱数据
        num_workers=0, #用0个子进程
    )

    # 先用一个CNN
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=2,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=2,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128,10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    model.train()
    optimizer.zero_grad() # 梯度清零

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    print("Image batch:", images.shape)
    print("Label batch:", labels.shape)
    print("Logits:", logits.shape)
    print("Loss:", loss.item())
    print("One training step   completed")

if __name__ == "__main__":
    main()

