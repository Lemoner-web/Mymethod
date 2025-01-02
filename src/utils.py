import torch
import pandas as pd

def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)  
    model.load_state_dict(checkpoint)

def load_data(dataset):
    """
    Load data files from the dataset folder.
    Args:
        dataset_path (str): Path to the dataset folder.
    Returns:
        dict: Dictionary containing DataFrames for each dataset file.
    """
    data = {
        "A_in_B": pd.read_csv(f"dataset/{dataset}/concept_hierarchy.txt", sep="\t", header=None, names=["A", "B"], dtype={'A':int, 'B':int}),
        "ArB": pd.read_csv(f"dataset/{dataset}/concept_relations.txt", sep="\t", header=None, names=["A", "r", "B"], dtype={'A':int, 'r':int, 'B':int}),
        "arb": pd.read_csv(f"dataset/{dataset}/individual_relations.txt", sep="\t", header=None, names=["a", "r", "b"], dtype={'a':int, 'r':int, 'b':int}),
        "a_in_A": pd.read_csv(f"dataset/{dataset}/individual_to_concept.txt", sep="\t", header=None, names=["a", "A"], dtype={'a':int, 'A':int}),
        # "valid": pd.read_csv(f"dataset/{dataset}/valid.txt", sep="\t", header=None, names=["a", "r", "b"]),
        # "test": pd.read_csv(f"dataset/{dataset}/test.txt", sep="\t", header=None, names=["a", "r", "b"]),
    }
    return data

def load_mappings(dataset):
    mappings = {'entity_to_idx':{},'concept_to_idx':{},'relation_to_idx':{}}
    with open(f'dataset/{dataset}/entities.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            entity, index = line.split('\t')
            mappings['entity_to_idx'][entity] = int(index)

    with open(f'dataset/{dataset}/concepts.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            concept, index = line.split('\t')
            mappings['concept_to_idx'][concept] = int(index)
    
    with open(f'dataset/{dataset}/relations.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            relation, index = line.split('\t')
            mappings['relation_to_idx'][relation] = int(index)

    return mappings

def get_nums(mappings):
    return len(mappings['entity_to_idx']), len(mappings['concept_to_idx']), len(mappings['relation_to_idx'])

def compute_l2_regularization(model, lambda_reg):
    l2_loss = 0.0
    for param in model.parameters():
        if param.requires_grad:  # 只对需要梯度的参数计算正则化
            l2_loss += torch.sum(param ** 2)
    return lambda_reg * l2_loss