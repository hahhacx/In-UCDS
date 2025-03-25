import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NGCF(nn.Module):
    def __init__(self, config, user_num, item_num, device):
        super(NGCF, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.device = device

        self.emb_dim = config['embed_size']
        self.layer_size = config['layer_size']
        self.n_layers = len(self.layer_size)

        self.adj_type = config['adj_type']
        self.alg_type = config['alg_type']
        self.n_fold = config['n_fold']
        self.node_dropout_flag = config['node_dropout_flag']
        self.node_dropout_rate = config['node_dropout_rate']
        self.decay = config['decay']

        self.norm_adj = self._create_norm_adj().to(device)

        self.user_embedding = nn.Embedding(self.user_num, self.emb_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.weights = self._init_weights()
        self.logistic = nn.Sigmoid()

    def _init_weights(self):
        weights = nn.ParameterDict()
        self.weight_size_list = [self.emb_dim] + self.layer_size

        for k in range(self.n_layers):
            weights[f'W_gc_{k}'] = nn.Parameter(torch.empty(
                self.weight_size_list[k], self.weight_size_list[k + 1]
            ))
            weights[f'b_gc_{k}'] = nn.Parameter(torch.zeros(1, self.weight_size_list[k + 1]))

            weights[f'W_bi_{k}'] = nn.Parameter(torch.empty(
                self.weight_size_list[k], self.weight_size_list[k + 1]
            ))
            weights[f'b_bi_{k}'] = nn.Parameter(torch.zeros(1, self.weight_size_list[k + 1]))

            nn.init.xavier_uniform_(weights[f'W_gc_{k}'])
            nn.init.xavier_uniform_(weights[f'W_bi_{k}'])

        return weights

    def _create_norm_adj(self):
        user_item_data = np.random.choice([0, 1], size=(self.user_num, self.item_num), p=[0.9, 0.1])
        user_item_matrix = torch.FloatTensor(user_item_data)

        adj_mat = torch.zeros(self.user_num + self.item_num, self.user_num + self.item_num)
        adj_mat[:self.user_num, self.user_num:] = user_item_matrix
        adj_mat[self.user_num:, :self.user_num] = user_item_matrix.T

        rowsum = adj_mat.sum(dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        norm_adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_mat), d_mat_inv_sqrt)
        return norm_adj

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.user_num + self.item_num) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.user_num + self.item_num
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(X[start:end, :])

        return A_fold_hat

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        if keep_prob == 1.0:
            return X

        dropout_rate = 1 - keep_prob
        dropout_X = F.dropout(X, p=dropout_rate, training=self.training)
        return dropout_X

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.user_num + self.item_num) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.user_num + self.item_num
            else:
                end = (i_fold + 1) * fold_len

            temp = X[start:end]
            n_nonzero_temp = temp.nonzero().size(0)
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout_rate, n_nonzero_temp))

        return A_fold_hat

    def _create_ngcf_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = torch.cat(temp_embed, 0)
            sum_embeddings = F.leaky_relu(
                torch.matmul(side_embeddings, self.weights[f'W_gc_{k}']) + self.weights[f'b_gc_{k}']
            )

            bi_embeddings = ego_embeddings * side_embeddings
            bi_embeddings = F.leaky_relu(
                torch.matmul(bi_embeddings, self.weights[f'W_bi_{k}']) + self.weights[f'b_bi_{k}']
            )

            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = F.dropout(ego_embeddings, p=self.node_dropout_rate, training=self.training)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num], 0)

        return u_g_embeddings, i_g_embeddings

    def forward(self, user_indices, item_indices):
        u_g_embeddings, i_g_embeddings = self._create_ngcf_embed()

        user_emb = u_g_embeddings[user_indices]
        item_emb = i_g_embeddings[item_indices]

        scores = torch.sum(user_emb * item_emb, dim=1)
        pred_scores = self.logistic(scores)

        return pred_scores

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)

        mf_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        regularizer = (self.user_embedding(users).norm(2).pow(2)
                       + self.item_embedding(pos_items).norm(2).pow(2)
                       + self.item_embedding(neg_items).norm(2).pow(2)) / 2

        emb_loss = self.decay * regularizer
        reg_loss = 0.0

        return mf_loss + emb_loss + reg_loss

    def inactive_embeddings_for_my_loss(self, samples: torch.LongTensor):
        u_g_embeddings, _ = self._create_ngcf_embed()
        device = u_g_embeddings.device
        samples = samples.to(device)
        return u_g_embeddings[samples[:, 0]]

    @torch.no_grad()
    def neighbor_embeddings_for_my_loss(self, samples: torch.LongTensor):
        u_g_embeddings, _ = self._create_ngcf_embed()
        device = u_g_embeddings.device
        samples = samples.to(device)
        return u_g_embeddings[samples[:, 1]]
