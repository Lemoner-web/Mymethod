import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import sys
sys.path.append('./src')
from data_loader import create_dataloaders
from utils import load_data,load_mappings, get_nums
from model import MultiTaskEmbeddingModel
from loss import CustomLoss
from trainer import train
from eval import evaluate
from validate import validate

from Dataset import TaskDataset
from torch.utils.data import DataLoader

def main(dataset):
    data = load_data(dataset)
    mappings = load_mappings(dataset)
    batch_size = 500
    train_data = create_dataloaders(data, mappings,batch_size=batch_size)
    num_individuals, num_concepts, num_relations = get_nums(mappings)
    embedding_dim = 1000
    learning_rate = 1e-3
    num_epochs = 100
    alpha = {"ArB": 1.0, "arb": 2.0, "a_in_A": 1.0, "A_in_B": 1.0}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    train_filter_ar = {}
    train_filter_rb = {}
    with open(f'dataset/{dataset}/train.txt', 'r') as f:
        for line in f.readlines():
            a,r,b = line.split('\t')
            a,r,b = int(a),int(r),int(b)
            if (r, b) not in train_filter_rb:
                train_filter_rb[(r, b)] = set()
            train_filter_rb[(r, b)].add(a)

            if (a, r) not in train_filter_ar:
                train_filter_ar[(a, r)] = set()
            train_filter_ar[(a, r)].add(b)
            
    test_data = pd.read_csv(f"dataset/{dataset}/test.txt", sep="\t", header=None, names=["a", "r", "b"], dtype={'a':int, 'r':int, 'b':int})
    # test_data = pd.read_csv(f"dataset/{dataset}/train.txt", sep="\t", header=None, names=["a", "r", "b"], dtype={'a':int, 'r':int, 'b':int})
    testDataset = TaskDataset(test_data, 'arb')
    testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)

    model = MultiTaskEmbeddingModel(num_concepts=num_concepts, num_individuals=num_individuals, num_relations=num_relations, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterions = {"nll": CustomLoss(), "bce": nn.BCELoss()}

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss, train_task_losses = train(model, train_data, optimizer, criterions, device, alpha)
        val_losses = validate(model, testDataloader, criterions, batch_size, device)

        # 打印损失
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        for task, loss in train_task_losses.items():
            print(f"  Task {task}: Loss = {loss:.4f}")
        print(f"Valid Loss = {val_losses:.4f}")
        if (epoch + 1) % 10 == 0:
            print("metrics: ", evaluate(model, testDataloader, train_filter_ar, train_filter_rb, device))
            torch.save(model.state_dict(), f"ckpt/{dataset}/model_{epoch}.pth")
        # 保存最好的模型
        # total_val_loss = val_losses
        # if total_val_loss < best_val_loss:
        #     best_val_loss = total_val_loss

    print("Training complete. Best model saved.")

if __name__ == "__main__":
    main('RadLex')


