import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from model import MultiTaskEmbeddingModel
from utils import load_model, load_mappings, get_nums
from Dataset import TaskDataset
from torch.utils.data import DataLoader

def forward_arb_batch(entity_embeddings, relation_matrices, idx, r_idx, candidate_indices, reverse=False):
    """
    批量计算 (a, r, ?) 或 (?, r, b) 的得分。
    Args:
        entity_embeddings: 所有实体的嵌入 [num_entities, dim]
        relation_matrices: 所有关系的嵌入 [num_relations, dim, dim]
        idx: 当前实体 a 的索引 (int) 或 b 的索引 (int)
        r_idx: 当前关系 r 的索引 (int)
        candidate_indices: 候选实体的索引列表 [batch_size]
        reverse: 是否计算 (?, r, b) 的得分 (默认为 False)
    Returns:
        scores: 每个候选实体的得分 [batch_size]
    """
    embedding_dim = entity_embeddings.size(1)
    r_matrix = relation_matrices[r_idx].view(-1, embedding_dim, embedding_dim) #[batch_size, dim, dim]
    candidate_embeddings = entity_embeddings[candidate_indices] #[num_individuals, dim]
    if reverse:
        # 计算 (?, r, b): Ar 与候选实体进行匹配
        b_embeddings = entity_embeddings[idx] #[batch_size, dim]
        candidates_expanded = candidate_embeddings.unsqueeze(0).expand(b_embeddings.size(0), -1, -1)
        candidate_r = torch.bmm(candidates_expanded, r_matrix) #[batch_size, nums, dim]     
        scores = torch.bmm(candidate_r, b_embeddings.unsqueeze(2)).squeeze(2)
        return scores
        
    else:
        # 计算 (a, r, ?): Ar 与候选实体进行匹配
        a_embeddings = entity_embeddings[idx]
        Ar = torch.bmm(a_embeddings.unsqueeze(1), r_matrix).squeeze(1) #[batch_size, dim]
        candidates_expanded = candidate_embeddings.unsqueeze(0).expand(a_embeddings.size(0), -1, -1) #[batch_size, nums, dim]
        scores = torch.bmm(candidates_expanded, Ar.unsqueeze(2)).squeeze(2)#[batch_size, nums]
    return scores

# 批量化处理训练集中存在的 (a, r, b)
def mask_scores(scores_ar, idx, r_idx, candidate_indices, train_set, reverse = False):
    """
    将出现在训练集中的 (a, r, b) 的分数置为 -inf。
    
    Args:
        scores_ar: [batch_size, num_candidates]，(a, r, ?) 的分数矩阵
        a_idx: [batch_size]，当前 batch 中的 a 索引
        r_idx: [batch_size]，当前 batch 中的 r 索引
        candidate_indices: [num_candidates]，候选实体的索引
        train_set: 训练集中所有 (a, r, b) 的集合
    
    Returns:
        scores_ar: 经过掩码处理的分数矩阵
    """
    # 将 candidate_indices 转为列表以便索引
    candidate_list = candidate_indices.tolist()
    if not reverse:
    # 遍历 batch
        for batch_idx, (a, r) in enumerate(zip(idx.tolist(), r_idx.tolist())):
            for candidate_idx, b in enumerate(candidate_list):
                # 如果 (a, r, b) 出现在训练集中，则将分数置为 -inf
                if (a, r, b) in train_set:
                    scores_ar[batch_idx, candidate_idx] = -1e6
    else:
        for batch_idx, (b, r) in enumerate(zip(idx.tolist(), r_idx.tolist())):
            for candidate_idx, a in enumerate(candidate_list):
                # 如果 (a, r, b) 出现在训练集中，则将分数置为 -inf
                if (a, r, b) in train_set:
                    scores_ar[batch_idx, candidate_idx] = -1e6
    
    return scores_ar



# def evaluate(model, test_dataloader, train_data, device):
    """
    在测试集上评估模型性能。
    Args:
        model: 已训练的模型
        test_dataloader: 测试集数据加载器
        train_data: 训练集中的三元组，格式为 [(a, r, b), ...]
        device: 设备（CPU 或 GPU）
    Returns:
        metrics: 包含 MR、MRR、Hits@1、Hits@3、Hits@10 的字典
    """
    model.eval()
    entity_embeddings = model.individual_embeddings.weight.detach().to(device)  # [num_entities, dim]
    relation_embeddings = model.relation_matrices.weight.detach().to(device)  # [num_relations, dim]

    train_set = set(train_data)

    ranks, reciprocal_ranks = [], []
    hits_at_k = {1: 0, 3: 0, 10: 0}

    with torch.no_grad():
        for batch in test_dataloader:
            a_idx, r_idx, b_idx, _ = batch
            a_idx, r_idx, b_idx = a_idx.to(device), r_idx.to(device), b_idx.to(device)
            candidate_indices = torch.arange(entity_embeddings.size(0), device=device)

            # (a, r, ?) 的分数
            scores_ar = forward_arb_batch(entity_embeddings, relation_embeddings, a_idx, r_idx, candidate_indices)
            # scores_ar = mask_scores(scores_ar, a_idx, r_idx, candidate_indices, train_set)
            target_score = scores_ar[torch.arange(scores_ar.size(0)), b_idx]
            ranks_ar = (scores_ar > target_score.unsqueeze(1)).sum(dim=1).cpu().numpy() + 1  # [batch_size]
            # 更新排名和指标
            ranks.extend(ranks_ar.tolist())
            reciprocal_ranks.extend((1.0 / ranks_ar).tolist())
            for k in hits_at_k.keys():
                hits_at_k[k] += (ranks_ar <= k).sum().item()

            # (?, r, b) 的分数
            scores_rb = forward_arb_batch(entity_embeddings, relation_embeddings, b_idx, r_idx, candidate_indices, reverse=True)
            # scores_rb = mask_scores(scores_ar, a_idx, r_idx, candidate_indices, train_set, True)
            target_score = scores_rb[torch.arange(scores_ar.size(0)), a_idx]
            ranks_rb = (scores_ar > target_score.unsqueeze(1)).sum(dim=1).cpu().numpy() + 1  # [batch_size]
            ranks.extend(ranks_rb.tolist())
            reciprocal_ranks.extend((1.0 / ranks_rb).tolist())
            for k in hits_at_k.keys():
                hits_at_k[k] += (ranks_rb <= k).sum().item()

    # 计算指标
    metrics = {
        "MR": sum(ranks) / len(ranks),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "Hits@1": hits_at_k[1] / len(ranks),
        "Hits@3": hits_at_k[3] / len(ranks),
        "Hits@10": hits_at_k[10] / len(ranks),
    }
    return metrics

def evaluate(model, test_dataloader, train_filter_ar, train_filter_rb, device):
    """
    在测试集上评估模型性能。
    Args:
        model: 已训练的模型
        test_dataloader: 测试集数据加载器
        train_filter_ar: {(a, r):{b}}
        device: 设备（CPU 或 GPU）
    Returns:
        metrics: 包含 MR、MRR、Hits@1、Hits@3、Hits@10 的字典
    """
    model.eval()
    ranks, reciprocal_ranks = [], []
    hits_at_k = {1: 0, 3: 0, 10: 0}

    with torch.no_grad():
        for batch in test_dataloader:
            a_idx, r_idx, b_idx, _ = batch
            a_idx, r_idx, b_idx = a_idx.to(device), r_idx.to(device), b_idx.to(device)

            # (a, r, ?) 的分数
            scores_ar = model.score_all(a_idx, r_idx, None, train_filter_ar, None)
            # scores_ar = mask_scores(scores_ar, a_idx, r_idx, candidate_indices, train_set)
            target_score = scores_ar[torch.arange(scores_ar.size(0)), b_idx]
            ranks_ar = (scores_ar > target_score.unsqueeze(1)).sum(dim=1).cpu().numpy() + 1  # [batch_size]
            # 更新排名和指标
            ranks.extend(ranks_ar.tolist())
            reciprocal_ranks.extend((1.0 / ranks_ar).tolist())
            for k in hits_at_k.keys():
                hits_at_k[k] += (ranks_ar <= k).sum().item()

            # (?, r, b) 的分数
            scores_rb = model.score_all(None, r_idx, b_idx, None, train_filter_rb)
            # scores_rb = mask_scores(scores_ar, a_idx, r_idx, candidate_indices, train_set, True)
            target_score = scores_rb[torch.arange(scores_ar.size(0)), a_idx]
            ranks_rb = (scores_ar > target_score.unsqueeze(1)).sum(dim=1).cpu().numpy() + 1  # [batch_size]
            ranks.extend(ranks_rb.tolist())
            reciprocal_ranks.extend((1.0 / ranks_rb).tolist())
            for k in hits_at_k.keys():
                hits_at_k[k] += (ranks_rb <= k).sum().item()

    # 计算指标
    metrics = {
        "MR": sum(ranks) / len(ranks),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "Hits@1": hits_at_k[1] / len(ranks),
        "Hits@3": hits_at_k[3] / len(ranks),
        "Hits@10": hits_at_k[10] / len(ranks),
    }
    return metrics



# 示例用法
if __name__ == "__main__":
    # 假设你有以下内容
    dataset = 'RadLex'
    model_path = f'ckpt/{dataset}/best_model.pth'
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    mappings = load_mappings(dataset)
    num_individuals, num_concepts, num_relations = get_nums(mappings)
    embedding_dim = 50
    batch_size = 64
    model = MultiTaskEmbeddingModel(num_concepts=num_concepts, num_individuals=num_individuals, num_relations=num_relations, embedding_dim=embedding_dim).to(device)
    load_model(model, model_path, device)
    
    test_set = []
    train_set = [] 
    
    test_data = pd.read_csv(f"dataset/{dataset}/train.txt", sep="\t", header=None, names=["a", "r", "b"], dtype={'a':int, 'r':int, 'b':int})
    with open(f'dataset/{dataset}/train.txt', 'r') as f:
        for line in f.readlines():
            a,r,b = line.split('\t')
            train_set.append((int(a), int(r), int(b)))
    dataset = TaskDataset(test_data, 'arb')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 评估性能
    results = evaluate(model, dataloader, train_set, device)
    print("Evaluation Metrics:", results)
