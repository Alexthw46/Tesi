import os
from tqdm import tqdm
import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.data_utils import get_dataset
from inference_val import get_template
from src.models.L2CLIP import CLIPWithL2P

# ---- User settings (replace paths / names as needed) ----
data_dir = "../data"
dataset_name = "cifar100"
arch = "L2Clip"
clip_model = "ViT-B/16"
batch_size = 64
work_dir = ".."
epochs = 5
lr = 1e-3
weight_decay = 1e-6
num_workers = 6
prompt_pool_size = 50


# --------------------------------------------------------


def get_classnames(dataset):
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    raise AttributeError("The selected dataset does not expose class names via a `classes` attribute.")

def train_one_epoch(net, dataloader, optimizer, criterion, device):
    net.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", total=len(dataloader)):
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = net(images)
        loss = criterion(logits.float(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(net, dataloader, criterion, device):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            images, labels = batch[0], batch[1]
            images = images.to(device)
            labels = labels.to(device)

            logits = net(images)
            loss = criterion(logits.float(), labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    _, preprocess = clip.load(clip_model, device=device)

    train_dataset = get_dataset(data_dir=data_dir, dataset=dataset_name, train=True, transform=preprocess)
    val_dataset = get_dataset(data_dir=data_dir, dataset=dataset_name, train=False, transform=preprocess)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    print("Datasets loaded")
    classnames = get_classnames(train_dataset)
    template = get_template(dataset_name)

    net = CLIPWithL2P(device=device, classnames=classnames, template=template, clip_model=clip_model,
                      prompt_pool_size=50)
    print("Model loaded.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.prompt.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(work_dir, exist_ok=True)
    ckpt_dir = os.path.join(work_dir, "../checkpoints", clip_model)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_" + str(prompt_pool_size) + "_prompt.pt")
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(net, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(net, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
