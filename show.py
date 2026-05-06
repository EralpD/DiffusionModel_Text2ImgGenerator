import torch
import torchvision
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1,0.1)),
        transforms.ToTensor(),
        lambda x: x*2-1 # Scale between -1 and 1
])
dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)

cifar_ds = torchvision.datasets.CIFAR10(
    root='../../Datasets/data',
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

def show_samples(grid, title):
    plt.figure(figsize=(8,8))
    plt.imshow(grid.detach().cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.title(title)
    plt.figure(figsize=(8,8))
    for i in range(16):
        ax, fig = plt.subplot(4,4,i+1), plt.gcf()
        index = torch.randint(0, len(dataset), (1,)).item()
        img, label = dataset[index]
        ax.imshow(img.squeeze().numpy(), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.axis('off')
    plt.show()

def show_images(grid, title, axises=[]):

    if len(axises) == 0:
        plt.figure(figsize=(8,8))
        plt.imshow(grid.detach().cpu().permute(1, 2, 0), interpolation='bicubic')
        plt.axis('off')
        plt.title(title)
    else:

        fig, axes = plt.subplots(1, len(axises), figsize=(15, 5))

        for i, (axis, t) in enumerate(axises):
            axis = axis.permute(1, 2, 0).cpu().numpy()
            axes[i].imshow(axis, interpolation='bicubic')
            axes[i].set_title(t)
            axes[i].axis('off')


    # plt.figure(figsize=(8, 8))
    # for i in range(16):
    #     ax, fig = plt.subplot(4,4,i+1), plt.gcf()
    #     index = torch.randint(0, len(cifar_ds), (1,)).item()
    #     img, label = cifar_ds[index]
    #     img = img.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    #     ax.imshow(img.numpy(), interpolation='bicubic')
    #     ax.axis('off')

    plt.show()

if __name__ == "__main__":
    show_images()