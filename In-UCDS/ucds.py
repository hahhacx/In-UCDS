import torch
import torch.nn as nn

class InUCDS(nn.Module):
    def __init__(self, num_users, active_ids, inactive_ids, similarity_matrix, alpha_coef=1.1, tol=1e-6, max_iter=5):
        super(InUCDS, self).__init__()
        self.num_users = num_users
        self.alpha_coef = alpha_coef
        self.tol = tol
        self.max_iter = max_iter

        self.active_ids = active_ids
        self.inactive_ids = inactive_ids

        self.similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)

    def compute_affinity_matrix(self, user_ids, target_idx):
        user_ids_tensor = torch.tensor(user_ids, dtype=torch.long)

        similarity_submatrix = self.similarity_matrix[user_ids_tensor][:, user_ids_tensor]
        similarity_submatrix.fill_diagonal_(0)
        similarity_submatrix = (similarity_submatrix + similarity_submatrix.T) / 2
        similarity_submatrix = similarity_submatrix - similarity_submatrix.min()
        similarity_submatrix = similarity_submatrix / (similarity_submatrix.max() + 1e-6)


        mask = torch.ones(len(user_ids), dtype=torch.float32)
        mask[target_idx] = 0
        I_S = torch.diag(mask)

        similarity_submatrix = torch.cat([similarity_submatrix[:target_idx], similarity_submatrix[target_idx + 1:]],
                                         dim=0)
        similarity_submatrix = torch.cat(
            [similarity_submatrix[:, :target_idx], similarity_submatrix[:, target_idx + 1:]], dim=1)

        if similarity_submatrix.numel() == 0:
            lambda_max = 1e-3
        else:
            lambda_max = max(torch.trace(similarity_submatrix) / similarity_submatrix.shape[0], 1e-3)

        alpha = self.alpha_coef * lambda_max

        result = self.similarity_matrix[user_ids_tensor][:, user_ids_tensor]
        result.fill_diagonal_(0)
        min_result = result.min()
        max_result = result.max()
        if max_result > min_result:
            result = (result - min_result) / (max_result - min_result)
        result -= alpha * I_S

        return result

    def replicator_dynamics(self, B, init_x):
        x = init_x.clone()
        toll = self.tol
        max_iter = self.max_iter

        for _ in range(max_iter):
            x_old = x.clone()
            x = x * (B @ x)
            x /= torch.norm(x, p=2, dim=0).detach()

            if torch.norm(x - x_old) < toll:
                break
        return x

    def extract_dominant_set(self, user_ids, target_user_id, neighbor_num):
        target_idx = user_ids.index(target_user_id)
        B = self.compute_affinity_matrix(user_ids, target_idx)

        init_x = torch.ones(len(user_ids), dtype=torch.float32) * 1e-6
        init_x[target_idx] = 1.0
        x = self.replicator_dynamics(B, init_x)

        sorted_x, indices = torch.sort(x, descending=True)

        dominant_ids = []
        for idx in indices:
            user_id = user_ids[idx.item()]
            if user_id != target_user_id:
                dominant_ids.append(user_id)
            if len(dominant_ids) == neighbor_num:
                break

        return torch.tensor(dominant_ids[:neighbor_num], dtype=torch.long)

    def generate_dominant_sets_for_all_inactive(self, neighbor_num):
        samples = []
        for inactive_user_id in self.inactive_ids:
            dominant_set = self.extract_dominant_set([inactive_user_id] + self.active_ids, inactive_user_id,
                                                     neighbor_num)
            for neighbor_id in dominant_set.tolist():
                samples.append([inactive_user_id, neighbor_id])

        return torch.tensor(samples, dtype=torch.long)

