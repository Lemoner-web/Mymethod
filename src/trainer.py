import torch
import torch.nn as nn
from utils import compute_l2_regularization
# def train(model, dataloaders, optimizer, criterions, device, alpha):
#     model.train()
#     total_loss = 0.0
#     task_losses = {task: 0 for task in alpha.keys()}  # 每个任务的损失
#     count = 0
#     for batch in dataloaders:
#         count += 1
#         optimizer.zero_grad()
#         batch_loss = 0.0
#         for task, data in batch.items():
#             # 将数据移动到GPU或CPU
#             if task == "ArB":
#                 A, r, B, label = [x.to(device) for x in data]
#                 # 三元组任务：计算预测值
#                 score = model.forward_ArB(A, r, B)
#                 loss = criterions['nll'](score, label)
#             elif task == "arb":
#                 a, r, b, label = [x.to(device) for x in data]
#                 # 处理 (a, r, b) 任务
#                 score = model.forward_arb(a, r, b)
#                 loss = criterions['nll'](score, label)
#             elif task == "A_in_B":
#                 A, B, label = [x.to(device) for x in data]
#                 # 处理 a 属于 A 的任务
#                 pred = model.forward_A_in_B(A, B)
#                 loss = criterions['bce'](pred.squeeze(1), label)
#             elif task == "a_in_A":
#                 a, A, label = [x.to(device) for x in data]
#                 # 处理 A 属于 B 的任务
#                 pred = model.forward_a_in_A(a, A)
#                 loss = criterions['bce'](pred.squeeze(1), label)
#             else:
#                 raise ValueError(f"Unknown task: {task}")

#             task_losses[task] += loss.item()
#             batch_loss += alpha[task] * loss
#         reg_loss = compute_l2_regularization(model, 1e-2)
#         total_batch_loss = batch_loss + reg_loss
#         total_batch_loss.backward()
#         optimizer.step()

#         total_loss += total_batch_loss.item()

#     avg_loss = total_loss / count
#     for task in task_losses:
#         task_losses[task] /= count

#     return avg_loss, task_losses

def train(model, dataloaders, optimizer, criterions, device):
    model.train()  # 设置模型为训练模式
    total_losses = {task: 0 for task in dataloaders.keys()}  # 每个任务的损失
    for task, dataloader in dataloaders.items():
        for batch in dataloader:
            # batch_loss = 0
            # 将数据移动到GPU或CPU
            if task == "ArB":
                A, r, B, label = [x.to(device) for x in batch]
                # 三元组任务：计算预测值
                score = model.forward_ArB(A, r, B)
                loss = criterions['nll'](score, label)
            elif task == "arb":
                a, r, b, label = [x.to(device) for x in batch]
                # 处理 (a, r, b) 任务
                score = model.forward_arb(a, r, b)
                loss = criterions['nll'](score, label)
            elif task == "A_in_B":
                A, B, label = [x.to(device) for x in batch]
                # 处理 a 属于 A 的任务
                pred = model.forward_A_in_B(A, B)
                loss = criterions['bce'](pred.squeeze(1), label)
            elif task == "a_in_A":
                a, A, label = [x.to(device) for x in batch]
                # 处理 A 属于 B 的任务
                pred = model.forward_a_in_A(a, A)
                loss = criterions['bce'](pred.squeeze(1), label)
            else:
                raise ValueError(f"Unknown task: {task}")
            # reg_loss = compute_l2_regularization(model, 1e-2)
            # batch_loss = reg_loss + loss
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            total_losses[task] += loss.item()

    # 平均每个任务的损失
    for task in total_losses:
        total_losses[task] /= len(dataloaders[task])

    return total_losses