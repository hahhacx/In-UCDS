import torch
import torch.nn as nn

class NeuMF(torch.nn.Module):
    def __init__(self, config, user_num, item_num, device):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = user_num
        self.num_items = item_num
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']
        self.device = device
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)
        self.ac_func = nn.ReLU()
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)


        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)


        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating


    def inactive_embeddings_for_my_loss(self, samples: torch.LongTensor):
        device = self.embedding_user_mlp.weight.device
        samples = samples.to(device)
        return torch.cat([self.embedding_user_mlp(samples[:, 0]), self.embedding_user_mf(samples[:, 0])])

    @torch.no_grad()
    def neighbor_embeddings_for_my_loss(self, samples: torch.LongTensor):
        device = self.embedding_user_mlp.weight.device
        samples = samples.to(device)
        return torch.cat([self.embedding_user_mlp(samples[:, 1]), self.embedding_user_mf(samples[:, 1])])

