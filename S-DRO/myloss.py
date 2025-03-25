import torch

class myLoss(torch.nn.Module):
    def __init__(self, l2):
        super().__init__()
        self.l2 = l2

    def forward(self, inactive_embeddings, neighbor_embeddings):
        # return self.l2 * torch.sum(torch.square(inactive_embeddings - neighbor_embeddings))
        return self.l2 * torch.sum(torch.square(inactive_embeddings - neighbor_embeddings))



