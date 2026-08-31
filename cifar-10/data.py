import torch
import torchvision
import torchvision.transforms as transforms

def load_data(batch_size):
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    traindataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_cifar)
    train_set = torch.utils.data.DataLoader(traindataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testdataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_cifar)
    test_set = torch.utils.data.DataLoader(testdataset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    return traindataset, train_set, testdataset, test_set
