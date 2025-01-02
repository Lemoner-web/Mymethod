import torch
from torch.utils.data import Dataset

class TaskDataset(Dataset):
    def __init__(self, dataframe, task_type):
        """
        :param dataframe: 包含数据的 pandas.DataFrame
        :param task_type: 任务类型 ("ArB", "arb", "a_in_A", "A_in_B")
        """
        self.data = dataframe
        self.task_type = task_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.task_type == "ArB":
            # 返回 (A, r, B, label)
            return (
                torch.tensor(row["A"], dtype=torch.long),
                torch.tensor(row["r"], dtype=torch.long),
                torch.tensor(row["B"], dtype=torch.long),
                torch.tensor(row["label"], dtype=torch.float)
            )
        elif self.task_type == "arb":
            # 返回 (a, r, b, label)
            try:
                return (
                    torch.tensor(row["a"], dtype=torch.long),
                    torch.tensor(row["r"], dtype=torch.long),
                    torch.tensor(row["b"], dtype=torch.long),
                    torch.tensor(row["label"], dtype=torch.float)
                )
            except:
                return (
                    torch.tensor(row["a"], dtype=torch.long),
                    torch.tensor(row["r"], dtype=torch.long),
                    torch.tensor(row["b"], dtype=torch.long),
                    torch.tensor(1.0, dtype = torch.float)
                )
        elif self.task_type == "a_in_A":
            # 返回 (a, A, label)
            return (
                torch.tensor(row["a"], dtype=torch.long),
                torch.tensor(row["A"], dtype=torch.long),
                torch.tensor(row["label"], dtype=torch.float)
            )
        elif self.task_type == "A_in_B":
            # 返回 (A, B, label)
            return (
                torch.tensor(row["A"], dtype=torch.long),
                torch.tensor(row["B"], dtype=torch.long),
                torch.tensor(row["label"], dtype=torch.float)
            )
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
