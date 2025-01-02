import torch
import torch.nn as nn
def validate(model, dataloader, criterions, batch_size, device):
    model.eval()
    total_loss = 0
    count = 0
    for batch in dataloader:
        a, r, b, label = [x.to(device) for x in batch]
        # 处理 (a, r, b) 任务
        score = model.forward_arb(a, r, b)
        loss = criterions['nll'](score, label)
        count += 1
        total_loss += loss.item()

    return total_loss / count