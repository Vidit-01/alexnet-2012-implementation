import torch
ckpt = torch.load("results/alexnet-spatialcrop.pth",map_location="cpu")
print(ckpt.keys())
