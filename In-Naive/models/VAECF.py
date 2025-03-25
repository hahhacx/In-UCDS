import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class PolyLinear(nn.Module):
    def __init__(self, layer_config: list, activation_fn, output_fn=None, input_dropout=None):
        super().__init__()
        assert len(layer_config) > 1

        self.layer_config = layer_config
        self.activation_fn = activation_fn
        self.output_fn = output_fn
        self.n_layers = len(layer_config) - 1

        layer_dict = OrderedDict()

        if input_dropout is not None:
            layer_dict["input_dropout"] = nn.Dropout(p=input_dropout)

        for i, (d1, d2) in enumerate(zip(layer_config[:-1], layer_config[1:])):
            layer_dict[f"linear_{i}"] = nn.Linear(d1, d2)
            if i < self.n_layers - 1:
                layer_dict[f"activation_{i}"] = activation_fn

        if self.output_fn is not None:
            layer_dict["output_activation"] = self.output_fn

        self.layers = nn.Sequential(layer_dict)

    def forward(self, x):
        return self.layers(x)


class VAECF(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, config, user_num, item_num, device):
        super(VAECF, self).__init__()
        self.num_users = user_num
        self.num_items = item_num
        self.latent_size = config['num_feat']
        self.device = device
        self.total_anneal_steps = config['total_anneal_steps']
        self.anneal_cap = config['anneal_cap']
        self.update_count = 0
        self.activation_fn = nn.Tanh()

        self.q_layers = PolyLinear([self.latent_size, 600, self.latent_size * 2], activation_fn=self.activation_fn)

        self.p_layers = PolyLinear([self.latent_size, 600, self.num_items], activation_fn=self.activation_fn)

        self.drop = nn.Dropout(config['dropout'])
        self.logistic = nn.Sigmoid()
        self.init_weights()

        self.user_embeddings = nn.Embedding(self.num_users, self.latent_size)

    def init_weights(self):
        for layer in self.q_layers.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, mean=0.0, std=0.001)

        for layer in self.p_layers.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, mean=0.0, std=0.001)

    def encode(self, user_vector):
        h = self.drop(F.normalize(user_vector.float()))
        h = self.q_layers(h)
        mu, logvar = h[:, :self.latent_size], h[:, self.latent_size:]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return z, KLD

    def decode(self, z):
        return self.p_layers(z)

    def forward(self, user_indices, item_indices):
        user_vectors = self.user_embeddings(user_indices)

        z, KLD = self.encode(user_vectors)
        scores = self.decode(z)
        pred_scores = scores.gather(1, item_indices.unsqueeze(1)).squeeze(1)
        pred_scores = self.logistic(pred_scores)

        return pred_scores

    def loss_function(self, recon_x, x, KLD, anneal=1.0):
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        return BCE + anneal * KLD

    def fit_epoch(self, device, train_loader, optimizer):
        self.train()
        total_loss, batch_count = 0, 0
        for user_indices, item_indices, _ in train_loader:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)

            anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps) if self.total_anneal_steps > 0 else self.anneal_cap

            self.zero_grad()
            recon_batch, KLD = self(user_indices, item_indices)
            loss = self.loss_function(recon_batch, user_indices, KLD, anneal)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            self.update_count += 1

        return total_loss, batch_count

    def inactive_embeddings_for_my_loss(self, samples: torch.LongTensor):
        device = self.user_embeddings.weight.device
        samples = samples.to(device)
        return self.user_embeddings(samples[:, 0])

    @torch.no_grad()
    def neighbor_embeddings_for_my_loss(self, samples: torch.LongTensor):
        device = self.user_embeddings.weight.device
        samples = samples.to(device)
        return self.user_embeddings(samples[:, 1])
