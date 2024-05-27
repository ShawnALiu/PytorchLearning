import torch.optim

from linear.linear_model import LinearModel
from dataset import generate_data


def main():
    epochs = 1000
    x_train, y_train = generate_data.generate_linear_data()
    y_train = torch.tensor(y_train, dtype=torch.float32)

    model = LinearModel()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, weight_decay=1e-2, momentum=0.)
    criterion = torch.nn.MSELoss()

    # 训练
    for epoch in range(epochs):
        loss = train(model, optimizer, criterion, x_train, y_train)
        print('epoch={}, loss={:.6f}'.format(epoch, loss))
    show_params(model)


def train(model, optimizer, criterion, x_train, y_train):

    input = torch.from_numpy(x_train)
    output = model(input)
    loss = criterion(output, y_train)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# 通过 named_parameters() 查看模型的可训练的参数
def show_params(model):
    for parameter in model.named_parameters():
        print(parameter)




if __name__ == '__main__':
    main()
