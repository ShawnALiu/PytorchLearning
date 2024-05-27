import torch


# 保存模型参数
def save_model(model, path, save_type=0):
    # 保存网络结构与参数
    if save_type == 1:
        torch.save(model, path)
    else:
        # 仅保存模型参数
        torch.save(model.state_dict(), path)


# 模型加载训练好的参数，用于推理
def load_data(model, path, save_type=0):
    if save_type == 1:
        model = torch.load(path)
    else:
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    model.eval()
    return model




