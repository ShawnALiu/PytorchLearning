import torch
from torch import nn

# 1.必须继承 nn.Module 类
class LinearModel(nn.Module):

    # 2.重写 __init__() 方法
    # 常来把有需要学习的参数的层放到构造函数中
    def __init__(self):
        # 必须调用父类的构造方法才可以
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    # 3.forward() 是必须重写的方法
    def forward(self, input):
        return (input * self.weight) + self.bias

