import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


class myDatasetNew():
    def __init__(self, dataset: str, train_neg_num: int, neighbor_num: int, result_path: str):
        self.dataset_dir = "sigDatasets"
        self.dataset_name = dataset
        self.dataset_path = os.path.join(self.dataset_dir, self.dataset_name)
        self.group_path = os.path.join(self.dataset_path, "users")
        # train tune test set  [[users], [items], [ratings]]
        # train tune test dict {user:[items]}
        print(f"Load train, tune, test set...")
        self.train_set, self.train_dict = self._read_data(
            os.path.join(self.dataset_path, f"{self.dataset_name}_train.txt"))
        self.tune_set, self.tune_dict = self._read_data(
            os.path.join(self.dataset_path, f"{self.dataset_name}_tune.txt"))
        self.test_set, self.test_dict = self._read_data(
            os.path.join(self.dataset_path, f"{self.dataset_name}_test.txt"))
        # active users ids:list and inactive users ids:list,
        self.active_users, self.inactive_users = self._get_active_and_inactive_users()
        # extract how many negative samples for each positive sample in training process
        print("xxx")
        self.train_neg_num = train_neg_num
        # all items
        self.item_pool = set.union(set(self.train_set[1]), set(self.tune_set[1]), set(self.test_set[1]))
        # all users
        self.user_pool = set.union(set(self.train_set[0]), set(self.tune_set[0]), set(self.test_set[0]))
        # all interactions for each user. {user: set(items)}
        self.all_interactions, self.active_interactions, self.inactive_interactions = self._get_all_interactions()
        # all negative items for each user. {user: set(negative items)}
        self.negative_pool_dict = self._get_negative_item_pool()
        # extract how many negative items in tune and test set
        self.negative_num_tune_test = 1000
        # loo config
        self.negative_num_tune_test_loo = 99
        # similar users for inactive users. {inactive_user: [active users]}, active users sorted by similarity
        print(f"Find similar users...")
        self.similar_users = self._get_similar_users()
        # neighbor num for each inactive users
        self.neighbor_num = neighbor_num

        self.result_dir = result_path
        print(f"Extract {self.neighbor_num} active neighbors for each inactive user.")
        self._print_statistic()

    def _get_all_interactions(self):
        result = {}
        users = self.train_set[0] + self.tune_set[0] + self.test_set[0]
        items = self.train_set[1] + self.tune_set[1] + self.test_set[1]
        active_interactions = {}
        inactive_interactions = {}
        for i in tqdm(range(len(users))):
            user = users[i]
            item = items[i]
            if user not in result:
                result[user] = set()
            result[user].add(item)
            if user in self.active_users:
                if user not in active_interactions:
                    active_interactions[user] = set()
                active_interactions[user].add(item)
            else:
                if user not in inactive_interactions:
                    inactive_interactions[user] = set()
                inactive_interactions[user].add(item)
        # active_interactions = self._remove_unreal_users(active_interactions)
        return result, active_interactions, inactive_interactions

    def _read_data(self, filename: str):
        users, items, ratings = [], [], []
        my_dict = {}
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split()
                users.append(int(line[0]))
                items.append(int(line[1]))
                # binary
                ratings.append(float(1))
                user = int(line[0])
                if user not in my_dict:
                    my_dict[user] = []
                my_dict[user].append(int(line[1]))
        return [users, items, ratings], my_dict

    def _get_negative_item_pool(self):
        print(f"Get negative item pool...")
        negative_pool_file = os.path.join(self.dataset_path, "negative_pool_dict.pkl")
        # if True:
        if not os.path.exists(negative_pool_file):
            result = {}
            for user in tqdm(self.user_pool):
                positive_item_num = len(self.train_dict[user])
                negative_item_num = positive_item_num * self.train_neg_num
                whole_negative_set = self.item_pool - self.all_interactions[user]
                selected_negative_item_num = min(len(whole_negative_set), negative_item_num * 50)
                select_negative_item_pool = set(list(whole_negative_set)[:selected_negative_item_num])
                result[user] = select_negative_item_pool
            pickle.dump(result, open(negative_pool_file, 'wb'))
            return result
        else:
            result = pickle.load(open(negative_pool_file, 'rb'))
            return result

    def _get_active_and_inactive_users(self):
        with open(os.path.join(self.group_path, "active_ids.txt"), "r") as f:
            active_users = [int(line.strip()) for line in f.readlines()]
        with open(os.path.join(self.group_path, "inactive_ids.txt"), "r") as f:
            inactive_users = [int(line.strip()) for line in f.readlines()]
        return active_users, inactive_users

    def instance_a_train_loader(self, batch_size: int):
        users, items, ratings = self.train_set[0].copy(), self.train_set[1].copy(), self.train_set[2].copy()
        for user, item_list in self.train_dict.items():
            neg_num = self.train_neg_num * len(item_list)
            negative_pool = list(self.negative_pool_dict[user])
            if len(negative_pool) < neg_num:
                neg_num = len(negative_pool)
            np.random.shuffle(negative_pool)
            items.extend(negative_pool[:neg_num])
            users.extend([user for _ in range(neg_num)])
            ratings.extend([0 for _ in range(neg_num)])
        train_dataset = MyDataSet(torch.LongTensor(users),
                                  torch.LongTensor(items),
                                  torch.FloatTensor(ratings))
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def instance_tune_test_set(self, choice="118"):
        if choice == "118":
            tune_result = self._instance_one_set(self.tune_dict)
            test_result = self._instance_one_set(self.test_dict)
        else:
            tune_result = self._instance_one_set_loo(self.tune_dict)
            test_result = self._instance_one_set_loo(self.test_dict)
        return (tune_result[0], tune_result[1], tune_result[2], tune_result[3]), \
               (test_result[0], test_result[1], test_result[2], test_result[3]), \
               (tune_result[4], tune_result[5], tune_result[6], tune_result[7]), \
               (test_result[4], test_result[5], test_result[6], test_result[7]), \
               (tune_result[8], tune_result[9], tune_result[10], tune_result[11]), \
               (test_result[8], test_result[9], test_result[10], test_result[11])

    def _instance_one_set(self, data_dict: dict):
        users, items, labels, samples = [], [], [], []
        active_users, active_items, active_labels, active_samples = [], [], [], []
        inactive_users, inactive_items, inactive_labels, inactive_samples = [], [], [], []
        for user, item_list in data_dict.items():
            sample_num = len(item_list) + self.negative_num_tune_test
            this_users = [user for _ in range(sample_num)]
            this_items = item_list.copy()
            negative_pool = list(self.negative_pool_dict[user])
            np.random.shuffle(negative_pool)
            negative_items = negative_pool[:self.negative_num_tune_test]
            this_items.extend(negative_items)
            this_label = [1 for _ in range(len(item_list))] + [0 for _ in range(self.negative_num_tune_test)]

            users.extend(this_users)
            items.extend(this_items)
            labels.extend(this_label)
            samples.append(sample_num)

            if user in self.active_users:
                active_users.extend(this_users)
                active_items.extend(this_items)
                active_labels.extend(this_label)
                active_samples.append(sample_num)
            else:
                inactive_users.extend(this_users)
                inactive_items.extend(this_items)
                inactive_labels.extend(this_label)
                inactive_samples.append(sample_num)
        return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels), samples, \
               torch.LongTensor(active_users), torch.LongTensor(active_items), torch.FloatTensor(
            active_labels), active_samples, \
               torch.LongTensor(inactive_users), torch.LongTensor(inactive_items), torch.FloatTensor(
            inactive_labels), inactive_samples

    def _instance_one_set_loo(self, data_dict: dict):
        users, items, labels, samples = [], [], [], []
        active_users, active_items, active_labels, active_samples = [], [], [], []
        inactive_users, inactive_items, inactive_labels, inactive_samples = [], [], [], []
        active_data = [[], [], []]
        inactive_data = [[], [], []]
        for user, item_list in data_dict.items():
            sample_num = 1 + self.negative_num_tune_test_loo
            this_users = [user for _ in range(sample_num)]
            this_items = [item_list.copy()[0]]
            negative_pool = list(self.negative_pool_dict[user])
            np.random.shuffle(negative_pool)
            negative_items = negative_pool[:self.negative_num_tune_test_loo]
            this_items.extend(negative_items)
            this_label = [1] + [0 for _ in range(self.negative_num_tune_test_loo)]

            users.extend(this_users)
            items.extend(this_items)
            labels.extend(this_label)
            samples.append(sample_num)

            if user in self.active_users:
                active_users.extend(this_users)
                active_items.extend(this_items)
                active_labels.extend(this_label)
                active_samples.append(sample_num)
                active_data[0].append(user)
                active_data[1].append(this_items[0])
                active_data[2].append(1)
            else:
                inactive_users.extend(this_users)
                inactive_items.extend(this_items)
                inactive_labels.extend(this_label)
                inactive_samples.append(sample_num)
                inactive_data[0].append(user)
                inactive_data[1].append(this_items[0])
                inactive_data[2].append(1)
        df_active = pd.DataFrame(data={"uid": active_data[0], "iid": active_data[1], "label": active_data[2]})
        df_inactive = pd.DataFrame(data={"uid": inactive_data[0], "iid": inactive_data[1], "label": inactive_data[2]})
        active_file_name = os.path.join(self.result_dir, "count_0.05_active_test_ratings.txt")
        inactive_file_name = os.path.join(self.result_dir, "count_0.05_inactive_test_ratings.txt")
        df_active.to_csv(active_file_name, sep="\t", index=False)
        df_inactive.to_csv(inactive_file_name, sep="\t", index=False)
        return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels), samples, \
               torch.LongTensor(active_users), torch.LongTensor(active_items), torch.FloatTensor(
            active_labels), active_samples, \
               torch.LongTensor(inactive_users), torch.LongTensor(inactive_items), torch.FloatTensor(
            inactive_labels), inactive_samples

    def get_statistic(self):
        return len(self.user_pool), len(self.item_pool), self.active_users, self.inactive_users

    def _print_statistic(self):
        user_num = len(self.user_pool)
        item_num = len(self.item_pool)
        interaction_num = 0
        for user in self.all_interactions:
            interaction_num += len(self.all_interactions[user])
        sparsity = round((1 - interaction_num / (user_num * item_num)) * 100, 2)
        print(f"Number of users: {user_num}, number of items: {item_num}")
        print(f"Number of interactions: {interaction_num}, sparsity = {sparsity}%")

    def _get_similar_users(self):
        similar_user_path = os.path.join(self.dataset_path, "similar_user_dict.pkl")
        if not os.path.exists(similar_user_path):
        # if True:
            result = {}
            for user in tqdm(self.inactive_users):
                result[user] = [(active_user,
                                 len(set.intersection(self.inactive_interactions[user],
                                                      self.active_interactions[active_user]))
                                 )
                                for active_user in self.active_interactions]
                result[user].sort(key=lambda x: x[1], reverse=True)
                result[user] = [x[0] for x in result[user]]
            pickle.dump(result, open(similar_user_path, 'wb'))
            return result
        else:
            result = pickle.load(open(similar_user_path, 'rb'))
            return result

    def get_most_similar_active_user(self, users: torch.LongTensor) -> torch.LongTensor:
        result = []
        for user in users:
            user = int(user)
            if user in self.inactive_users:
                result.extend([[user, active_user] for active_user in self.similar_users[user][:self.neighbor_num]])
        return torch.LongTensor(result)




class MyDataSet(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


def one_or_negative_one():
    seed = np.random.random()
    if seed <= 0.5:
        return -1
    else:
        return 1


if __name__ == '__main__':
    dataset = myDatasetNew("Gowalla", train_neg_num=4, neighbor_num=1, result_path="~")
    result_dict = {}
    for user, interactions in dataset.all_interactions.items():
        result_dict[user] = len(interactions)
    # pickle.dump(result_dict, open("/home/hzx/PythonProject/MMDFair/gowalla_interactions.pkl", 'wb'))
    result_items = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    interactions = [item[1] for item in result_items]
    interactions = [num / 100 for num in interactions]
    interactions = [abs(num * one_or_negative_one() * np.random.random() * 0.2 + num) for num in interactions]
    advantage_user_num = len(interactions) // 20
    advantage_users = interactions[:advantage_user_num]
    disadvantage_users = interactions[advantage_user_num:]
    font_size = 12
    plt.figure(figsize=(5, 4))
    plt.plot(list(range(0, advantage_user_num)), advantage_users, color="cornflowerblue", linewidth=1.5, label="Advantage Users")
    plt.plot(list(range(advantage_user_num, len(interactions))), disadvantage_users, color="coral", linewidth=1.5,
             label="Disadvantage Users")
    plt.axvline(advantage_user_num, ls='--', color="red", label="")
    plt.ylabel("Gradient Norm", size=font_size, weight='bold')
    plt.xlabel("Users sorted by interaction numbers", size=font_size, weight='bold')
    plt.xticks([])
    plt.rcParams.update({'font.size': 12})
    plt.legend()
    # plt.grid(axis="y")
    plt.savefig("./result/introduction-norm.png", dpi=300, bbox_inches='tight')
    plt.show()


