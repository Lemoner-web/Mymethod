import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, score, label):
        """
        计算自定义损失
        :param score: 模型的输出（logits），形状为 (batch_size,)
        :param label: 真实标签，值为 0 或 1，形状为 (batch_size,)
        :return: 损失值
        """
        # 将标签从 [0, 1] 转换为 [-1, 1]
        label_transformed = label * 2 - 1
        # 计算损失
        loss = torch.mean(torch.log(1 + torch.exp(-label_transformed * score)))
        return loss