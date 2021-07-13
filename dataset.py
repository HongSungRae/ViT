from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision



class MNISTData(Dataset):
    def __init__(self,usage,path='./mnist'):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self):
        pass


if __name__ == '__main__':
    torch.manual_seed(42)

    DOWNLOAD_PATH = './mnist'
    BATCH_SIZE_TRAIN = 128
    BATCH_SIZE_TEST = 256

    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.Grayscale(3),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = MNIST(DOWNLOAD_PATH, train=True, download=True, transform=transform_mnist)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_set = MNIST(DOWNLOAD_PATH, train=False, download=True, transform=transform_mnist)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

    img,y = next(iter(train_loader))
    print(img)
    print(img.shape)
    print(y)
    print(y.shape)
    print(len(train_loader),len(test_loader))