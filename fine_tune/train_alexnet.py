import numpy as np
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from dataset import generate_data


def main():
    epochs = 10
    alexnet = _get_model()
    train_dataloader = generate_data.generate_cifar10(shuffle=True)
    optimizer = torch.optim.SGD(alexnet.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()


    for epoch in range(1, epochs + 1):
        loss = train(epoch, train_dataloader, alexnet, optimizer, criterion, epochs)



def train(epoch, dataloader, model, optimizer, criterion, epochs):
    losses = []
    pbar = tqdm(dataloader, ncols=100)
    for step, item in enumerate(pbar, start=1):
        input = item[0]
        target = item[1]
        output = model(input)
        loss = criterion(output, target)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("[{}/{}]TRAIN: STEP_LOSS={:.5f}".format(epoch, epochs, loss.item()))

    return np.mean(losses)


@torch.no_grad()
def test(path):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)

    alexnet = torchvision.models.alexnet(pretrained=True)
    alexnet.eval()

    pred = alexnet(input_tensor).argmax()
    print('图片={}，预测的标签为={}'.format(path, pred))


# 修改模型，进行微调
def _get_model():
    alexnet = torchvision.models.alexnet(pretrained=True)
    print(alexnet)
    # (classifier): Sequential(
    #   ...
    #   (6): Linear(in_features=4096, out_features=1000, bias=True)
    # )

    # 将最后一层全连接层改为输出只有10维，因为cifar10只有10类
    fc_in_features = alexnet.classifier[6].in_features
    alexnet.classifier[6] = torch.nn.Linear(fc_in_features, 10)
    print(alexnet)
    return alexnet



if __name__ == '__main__':
    # _path = '../data/dog.jpg'
    # test(_path)

    main()
