import argparse
import importlib.util
import torch
from torch.utils.data import DataLoader

from utils.dataset import TinyImageNetVal
from utils.transforms import val_transform


def load_model(model_path, num_classes):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AlexNet(num_classes=num_classes)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model .py file")
    parser.add_argument("checkpoint", help="Path to saved .pth file")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # auto device fallback
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nUsing device: {device}")

    # load model architecture
    model = load_model(args.model_path, args.num_classes).to(device)

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint from: {args.checkpoint}")

    # build validation dataloader
    val_set = TinyImageNetVal(transform=val_transform)
    val_loader = DataLoader(val_set,
                            batch_size=args.bs,
                            shuffle=False,
                            num_workers=args.workers)

    # evaluate
    acc = evaluate(model, val_loader, device)

    print(f"\nValidation Accuracy: {acc * 100:.2f}%\n")


if __name__ == "__main__":
    main()
