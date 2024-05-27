import random

import numpy as np
from matplotlib import pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def generate_linear_data():
    w = 2
    b = 3
    xlim = [-10, 10]
    x_train = np.random.randint(low=xlim[0], high=xlim[1], size=30)
    y_train = [w * x + b + random.randint(0, 2) for x in x_train]
    plt.plot(x_train, y_train, 'bo')
    # plt.show()
    return x_train, y_train


def generate_cifar10():
    cifar10_dataset = torchvision.datasets.CIFAR10(root='../data',
                                                   train=False,
                                                   transform=transforms.ToTensor(),
                                                   target_transform=None,
                                                   download=True)
    # 取32张图片的tensor
    tensor_dataloader = DataLoader(dataset=cifar10_dataset, batch_size=256)
    for index, (img_tensor, label_tensor) in enumerate(tensor_dataloader):
        if index >= 1:
            break
        print(img_tensor.shape)
        grid_tensor = torchvision.utils.make_grid(img_tensor, nrow=16, padding=2)
        grid_img = transforms.ToPILImage()(grid_tensor)
        grid_img.show()


if __name__ == '__main__':
    # generate_linear_data()
    generate_cifar10()
