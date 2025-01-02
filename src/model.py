import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class MultiTaskEmbeddingModel(nn.Module):
    def __init__(self, num_concepts, num_individuals, num_relations, embedding_dim):
        super(MultiTaskEmbeddingModel, self).__init__()
        # Embedding for concepts, individuals, and relations
        self.num_individuals = num_individuals
        self.embedding_dim = embedding_dim
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)
        self.individual_embeddings = nn.Embedding(num_individuals, embedding_dim)
        self.relation_matrices = nn.Embedding(num_relations, embedding_dim * embedding_dim)

        # 对 concept_embeddings 和 individual_embeddings 进行 Xavier 初始化
        init.xavier_uniform_(self.concept_embeddings.weight)
        init.xavier_uniform_(self.individual_embeddings.weight)

        # 对 relation_matrices 进行初始化
        init.xavier_uniform_(self.relation_matrices.weight)

        # MLP for membership prediction (a ∈ A, A ⊆ B)
        self.mlp_a_in_A = nn.Sequential(nn.Linear(embedding_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.mlp_A_in_B = nn.Sequential(nn.Linear(embedding_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward_ArB(self, head, relation, tail):

        A = self.concept_embeddings(head)
        B = self.concept_embeddings(tail) 
        
        # 获取关系 r 的嵌入，并 reshape 为 [batch_size, dim, dim]
        r = self.relation_matrices(relation).view(-1, self.embedding_dim, self.embedding_dim)
        
        # 计算 A^T r B
        # 1. 计算 r B，结果 shape 为 [batch_size, dim]
        rB = torch.bmm(r, B.unsqueeze(-1)).squeeze(-1)
        # 2. 计算 A^T (r B)，结果 shape 为 [batch_size]
        score = torch.sum(A * rB, dim=1) / self.embedding_dim
        return score

    def forward_arb(self, head, relation, tail):
        A = self.individual_embeddings(head)
        B = self.individual_embeddings(tail)
        
        # 获取关系 r 的嵌入，并 reshape 为 [batch_size, dim, dim]
        r = self.relation_matrices(relation).view(-1, self.embedding_dim, self.embedding_dim)
        
        # 计算 A^T r B
        # 1. 计算 r B，结果 shape 为 [batch_size, dim]
        rB = torch.bmm(r, B.unsqueeze(-1)).squeeze(-1)
        # 2. 计算 A^T (r B)，结果 shape 为 [batch_size]
        score = torch.sum(A * rB, dim=1) / self.embedding_dim
        return score

    def forward_A_in_B(self, head, tail):
        h = self.concept_embeddings(head)
        t = self.concept_embeddings(tail)
        return torch.sigmoid(self.mlp_A_in_B(torch.cat([h, t], dim=-1)))

    def forward_a_in_A(self, head, tail):
        h = self.individual_embeddings(head)
        t = self.concept_embeddings(tail)
        return torch.sigmoid(self.mlp_a_in_A(torch.cat([h, t], dim=-1)))

    def score_all(self, A_idx=None, r_idx=None, B_idx=None, train_filter_ar=None, train_filter_rb=None):
        """
        对所有可能的 (?, r, B) 或 (A, r, ?) 进行打分。 B: [nums]
        """
        # A^T r B  [batch_size, nums]
        # 获取所有实体嵌入，shape 为 [num_entities, dim]
        all_embeddings = self.individual_embeddings.weight  # [num_entities, dim]

        # 获取当前关系嵌入，shape 为 [batch_size, dim, dim]
        r = self.relation_matrices(r_idx).view(-1, self.embedding_dim, self.embedding_dim)

        if A_idx is None:
            # 计算 (?, r, B): 对所有 A 进行打分
            B = self.individual_embeddings(B_idx)  # [batch_size, dim]
            # rB 的形状为 [batch_size, dim]
            rB = torch.bmm(r, B.unsqueeze(-1)).squeeze(-1)
            # 对每个实体 A 计算分数，结果为 [batch_size, num_entities]
            scores = torch.matmul(rB, all_embeddings.T)
            for i, (r_val, b_val) in enumerate(zip(r_idx.tolist(), B_idx.tolist())):
                if (r_val, b_val) in train_filter_rb:
                    invalid_a = train_filter_rb[(r_val, b_val)]
                    scores[i, list(invalid_a)] = -1e6

        elif B_idx is None:
            # 计算 (A, r, ?): 对所有 B 进行打分
            A = self.individual_embeddings(A_idx)  # [batch_size, dim]
            # rA 的形状为 [batch_size, dim]
            # r^T A
            rA = torch.bmm(r.transpose(1, 2), A.unsqueeze(-1)).squeeze(-1)
            # 对每个实体 B 计算分数，结果为 [batch_size, num_entities]
            scores = torch.matmul(rA, all_embeddings.T)
            for i, (a_val, r_val) in enumerate(zip(A_idx.tolist(), r_idx.tolist())):
                if (a_val, r_val) in train_filter_ar:
                    invalid_b = train_filter_ar[(a_val, r_val)]
                    scores[i, list(invalid_b)] = -1e6

        else:
            raise ValueError("Either A_idx or B_idx must be None to compute (?, r, B) or (A, r, ?).")

        return scores