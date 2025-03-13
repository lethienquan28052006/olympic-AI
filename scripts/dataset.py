import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, len(train_dataset.classes)

if __name__ == "__main__":
    train_loader, num_classes = get_data_loaders("D:\Olympic AI\data")
    print(f"Số lớp: {num_classes}")