import argparse
import importlib.util
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# === your utils ===
from utils.dataset import TinyImageNetTrain, TinyImageNetVal
# from utils.transforms import train_transform, val_transform

def load_transforms(path):
    spec = importlib.util.spec_from_file_location("transform_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train_transform, module.val_transform

def load_model(model_path, num_classes):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AlexNet(num_classes=num_classes)


def get_dataloaders(batch_size, num_workers,transforms):
    train_transform, val_transform = load_transforms(transforms)
    train_set = TinyImageNetTrain(transform=train_transform)
    val_set   = TinyImageNetVal(transform=val_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader



def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0

    start_time = time.time()

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch}/{total_epochs} [training]",
        leave=False
    )

    for batch_idx, (x, y) in progress:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        elapsed = time.time() - start_time
        batches_done = batch_idx + 1
        batches_total = len(loader)
        eta = elapsed / batches_done * (batches_total - batches_done)

        progress.set_postfix({
            "batch": f"{batches_done}/{batches_total}",
            "loss": f"{loss.item():.4f}",
            "eta": f"{eta:.1f}s"
        })

    return total_loss / len(loader)


def evaluate(model, loader, device, epoch, total_epochs):
    model.eval()
    correct, total = 0, 0

    start_time = time.time()

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch}/{total_epochs} [validation]",
        leave=False
    )

    with torch.no_grad():
        for batch_idx, (x, y) in progress:
            x, y = x.to(device), y.to(device)

            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            elapsed = time.time() - start_time
            batches_done = batch_idx + 1
            batches_total = len(loader)
            eta = elapsed / batches_done * (batches_total - batches_done)

            progress.set_postfix({
                "batch": f"{batches_done}/{batches_total}",
                "acc": f"{(correct/total):.4f}",
                "eta": f"{eta:.1f}s"
            })

    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="results/")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--transforms", default=os.path.join("utils","transforms_baseline.py"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Auto device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n✅ Using device: {device}")

    model = load_model(args.model_path, args.num_classes).to(device)

    train_loader, val_loader = get_dataloaders(args.bs, args.workers,args.transforms)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
    )

    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss = train_epoch(model, train_loader,
                                 optimizer, criterion, device,
                                 epoch, args.epochs)

        val_acc = evaluate(model, val_loader, device,
                           epoch, args.epochs)

        print(f"✅ Epoch {epoch} done | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": args.num_classes
            }, os.path.join(args.out, "best.pth"))
            print("💾 Saved new best model!")

    print(f"\n🎉 Training complete! Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
