import torch
import torchvision
from PIL import Image
from torchvision import transforms


def main():

    alexnet = torchvision.models.alexnet(pretrained=True)



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


if __name__ == '__main__':
    _path = '../data/dog.jpg'
    test(_path)
