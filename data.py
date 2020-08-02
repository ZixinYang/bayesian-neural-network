import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_download():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(),])),
        batch_size=128, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist-data/', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(),])),
        batch_size=128, shuffle=True)
    return train_loader, test_loader