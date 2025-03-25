import torch
import torch.nn as nn

class PMF(nn.Module):
    def __init__(self, config, user_num, item_num, device):
        super(PMF, self).__init__()
        self.num_users = user_num
        self.num_items = item_num
        self.num_feat = config['num_feat']
        self.device = device

        self.w_User = nn.Parameter(0.1 * torch.randn(user_num, self.num_feat, device=device))
        self.w_Item = nn.Parameter(0.1 * torch.randn(item_num, self.num_feat, device=device))

        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_vec = self.w_User[user_indices]
        item_vec = self.w_Item[item_indices]
        pred_scores = torch.sum(user_vec * item_vec, dim=1)
        pred_scores = self.logistic(pred_scores)

        return pred_scores


    def inactive_embeddings_for_my_loss(self, samples: torch.LongTensor):
        device = self.w_User.device
        samples = samples.to(device)
        return self.w_User[samples[:, 0]]

    @torch.no_grad()
    def neighbor_embeddings_for_my_loss(self, samples: torch.LongTensor):
        device = self.w_User.device
        samples = samples.to(device)
        return self.w_User[samples[:, 1]]