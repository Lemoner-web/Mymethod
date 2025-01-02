import random
import pandas as pd
from Dataset import TaskDataset
from torch.utils.data import DataLoader
from utils import load_data, load_mappings

def negative_sampling(data:pd.DataFrame, mappings, sample_size, seed):
    """
    Generate negative samples for a given dataset, ensuring they do not overlap with positive samples.
    Args:
        data (pd.DataFrame): DataFrame containing positive samples (e.g., a, r, b).
        mappings (dict): Mappings of entities, concepts, and relations to indices.
        sample_size (int): Number of negative samples to generate for each positive sample.
    Returns:
        list: List of tuples containing both positive and negative samples.
    """
    if seed is not None:
        random.seed(seed)

    all_entities = list(mappings["entity_to_idx"].values())
    all_concepts = list(mappings["concept_to_idx"].values())
    # 构建正样本集合
    positive_samples = set(data.itertuples(index=False, name=None))

    negative_samples = []
    for pos_sample in positive_samples:
        pos_sample = list(pos_sample)  # 将 tuple 转为 list 以便修改
        for _ in range(sample_size):
            neg_sample = pos_sample.copy()

            # 根据列数选择负采样策略
            if len(neg_sample) == 3:  # 三元组 (e.g., a, r, b) 或 (A, r, B)
                col_to_replace = random.choice([0, 2])  # 替换首列或尾列
                if col_to_replace == 0:  # 替换首列
                    neg_sample[0] = random.choice(all_entities if "a" in data.columns else all_concepts)
                elif col_to_replace == 2:  # 替换尾列
                    neg_sample[2] = random.choice(all_entities if "b" in data.columns else all_concepts)
            elif len(neg_sample) == 2:  # 二元组 (e.g., A subclassof B 或 A(a))
                col_to_replace = random.choice([0, 1])  # 替换任意一列
                if col_to_replace == 0:  # 替换首列
                    neg_sample[0] = random.choice(all_entities if "a" in data.columns else all_concepts)
                elif col_to_replace == 1:  # 替换尾列
                    neg_sample[1] = random.choice(all_concepts)

            # 确保生成的负样本不在正样本集合中
            max_attempts = 10
            attempts = 0
            while tuple(neg_sample) in positive_samples and attempts < max_attempts:
                if len(neg_sample) == 3:  # 三元组重新替换
                    col_to_replace = random.choice([0, 2])
                    if col_to_replace == 0:
                        neg_sample[0] = random.choice(all_entities if "a" in data.columns else all_concepts)
                    elif col_to_replace == 2:
                        neg_sample[2] = random.choice(all_entities if "b" in data.columns else all_concepts)
                elif len(neg_sample) == 2:  # 二元组重新替换
                    col_to_replace = random.choice([0, 1])
                    if col_to_replace == 0:
                        neg_sample[0] = random.choice(all_entities if "a" in data.columns else all_concepts)
                    elif col_to_replace == 1:
                        neg_sample[1] = random.choice(all_concepts)
                attempts += 1

            # 添加到负样本列表
            negative_samples.append(tuple(neg_sample))
    return pd.DataFrame(negative_samples, columns=data.columns)

def generate_negative_samples(data, mappings, sample_size=1, seed=None):
    """
    Generate negative samples for all tasks.
    Args:
        processed_data (dict): Preprocessed dataset.
        mappings (dict): Mappings of entities, concepts, and relations.
        sample_size (int): Number of negative samples to generate for each positive sample.
    Returns:
        dict: Dictionary containing positive and negative samples for each dataset.
    """
    negative_data = {
        "ArB": negative_sampling(data["ArB"], mappings, sample_size, seed),
        "arb": negative_sampling(data["arb"], mappings, sample_size, seed),
        "a_in_A": negative_sampling(data["a_in_A"], mappings, sample_size, seed),
        "A_in_B": negative_sampling(data["A_in_B"], mappings, sample_size, seed),
    }
    return negative_data

def construct_training_data(data, negative_data):
    training_data = {}
    for key in data.keys():
        # 加入正样本，标签为 1
        positive = data[key].copy()
        if positive.empty:
            continue
        positive["label"] = 1

        # 加入负样本，标签为 0
        negative = negative_data[key].copy()
        negative["label"] = 0

        # 合并正负样本并打乱
        combined = pd.concat([positive, negative]).sample(frac=1).reset_index(drop=True)
        training_data[key] = combined
    return training_data

# def create_dataloaders(data, mappings, seed=0, batch_size=64):

#     negative_data = generate_negative_samples(data, mappings, sample_size=1, seed=seed)
#     training_data = construct_training_data(data, negative_data)
#     dataloaders = {}
#     for task_type, dataframe in training_data.items():
#         dataset = TaskDataset(dataframe, task_type)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         dataloaders[task_type] = dataloader
#     iterators = {task: iter(loader) for task, loader in dataloaders.items()}
#     remaining_tasks = set(dataloaders.keys())
#     while remaining_tasks:
#         batch = {}
#         for task in list(remaining_tasks):  # Use list to avoid modifying the set during iteration
#             try:
#                 batch[task] = next(iterators[task])
#             except StopIteration:
#                 # Remove task from remaining_tasks when its data is exhausted
#                 remaining_tasks.remove(task)
#         if batch:  # Yield only if there is data in the batch
#             yield batch

def create_dataloaders(data, mappings, seed=0, batch_size=64):

    negative_data = generate_negative_samples(data, mappings, sample_size=1, seed=seed)
    training_data = construct_training_data(data, negative_data)
    dataloaders = {}
    for task_type, dataframe in training_data.items():
        dataset = TaskDataset(dataframe, task_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders[task_type] = dataloader
    return dataloaders

if __name__ == "__main__":
    dataset = 'RadLex'
    data = load_data(dataset)
    mappings = load_mappings(dataset)
    dataloaders = create_dataloaders(data, mappings)
    print(len(dataloaders))
    # batch = next(dataloaders)
    # print(batch)




