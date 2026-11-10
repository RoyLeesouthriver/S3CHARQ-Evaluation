"""CIFAR10 test split only; no augmentation or normalization."""
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def make_loader(root, batch_size, max_samples=0, workers=0, download=False):
    dataset = CIFAR10(root=str(root), train=False, download=download, transform=ToTensor())
    if max_samples:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=workers, drop_last=False)

