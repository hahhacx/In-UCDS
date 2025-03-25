import argparse
import numpy as np
import torch
import os
import config
from tqdm import tqdm
from sklearn.metrics import ndcg_score, f1_score, accuracy_score
from myloss import myLoss
from sigdatasets import myDatasetNew
import time
import pandas as pd
import random
import models.PMF as PMF
import models.NeuMF as NeuMF
import models.NGCF as NGCF
import models.VAECF as VAECF


SPLIT = "=" * 60
K = 10
L2 = 1e-5
fair_model_time = 0
eta=0.01
alpha=0.1
temperature=0.07


BEST_NDCG_ALL = 0
BEST_F1_ALL = 1
BEST_NDCG_ACTIVE = 2
BEST_F1_ACTIVE = 3
BEST_NDCG_INACTIVE = 4
BEST_F1_INACTIVE = 5
BEST_NDCG_UGF = 6
BEST_F1_UGF = 7
RESULT_USER = 0
RESULT_ITEM = 1
RESULT_SCORE = 2
RESULT_LABEL = 3

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--model', type=str, default="NeuMF", help="The trained model")
parser.add_argument('--cuda-index', type=int, default=1, help='train in which GPU')
parser.add_argument('--dataset', type=str, default="MovieLens", help='train in which dataset')
parser.add_argument('--split', type=str, default="count", help='how to split active users and inactive users')
parser.add_argument('--random-seed', type=int, default=42, help="The random seed of this program")
parser.add_argument('--neighbor_num', type=int, default=3, help='Extract how many neighbors for each inactive user')
parser.add_argument('--log', type=str, default='logs/{}'.format(parser.parse_args().model), help='log directory')
parser.add_argument('--model-time', type=int, default=0, help='The training time of the same model. ')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



#fair_model:S-DRO
def train_SDRO(epoch):
    print(SPLIT)
    print(f"Train epoch: {epoch}")
    fair_loss_list = [[], [], [], []]
    active_loss_avg = 0.0
    inactive_loss_avg = 0.0

    for batch_idx, data in tqdm(enumerate(sample_generator)):
        users, items, rating = data[0].to(device), data[1].to(device), data[2].to(device)

        active_users, active_items, active_ratings = [], [], []
        inactive_users, inactive_items, inactive_ratings = [], [], []

        for i in range(len(users)):
            if users[i].item() in dataset.active_users:
                active_users.append(users[i])
                active_items.append(items[i])
                active_ratings.append(rating[i])
            elif users[i].item() in dataset.inactive_users:
                inactive_users.append(users[i])
                inactive_items.append(items[i])
                inactive_ratings.append(rating[i])

        if len(active_users) > 0:
            active_users = torch.stack(active_users).to(device)
            active_items = torch.stack(active_items).to(device)
            active_ratings = torch.stack(active_ratings).to(device)
        if len(inactive_users) > 0:
            inactive_users = torch.stack(inactive_users).to(device)
            inactive_items = torch.stack(inactive_items).to(device)
            inactive_ratings = torch.stack(inactive_ratings).to(device)

        model_fair.train()
        fair_model_opt.zero_grad()

        if len(active_users) > 0:
            active_predictions = model_fair(active_users, active_items).view(-1)
            active_loss = bce_loss(active_predictions, active_ratings.float())
        else:
            active_loss = torch.tensor(0.0, device=device)

        if len(inactive_users) > 0:
            inactive_predictions = model_fair(inactive_users, inactive_items).view(-1)
            inactive_loss = bce_loss(inactive_predictions, inactive_ratings.float())
        else:
            inactive_loss = torch.tensor(0.0, device=device)

        active_loss_avg = (1 - alpha) * active_loss_avg + alpha * active_loss.item()
        inactive_loss_avg = (1 - alpha) * inactive_loss_avg + alpha * inactive_loss.item()

        loss_vector = torch.tensor([active_loss_avg, inactive_loss_avg], device=device)

        scaled_loss_vector = loss_vector / (loss_vector.max() + 1e-8)
        weights = torch.exp(eta * scaled_loss_vector / temperature)
        weights[1] *= 1 + (inactive_loss_avg / (active_loss_avg + 1e-8))
        weights /= weights.sum()

        weights = torch.softmax(weights * 5, dim=0)

        sdro_weighted_loss = weights[0] * active_loss + weights[1] * inactive_loss

        sdro_weighted_loss.backward()
        fair_model_opt.step()

        fair_loss_list[0].append(active_loss.item() + inactive_loss.item())
        fair_loss_list[1].append(active_loss.item())
        fair_loss_list[2].append(inactive_loss.item())
        fair_loss_list[3].append(sdro_weighted_loss.item())

    print(f"Total Loss: {round(np.mean(fair_loss_list[0]), 4)}, "
          f"Active Loss: {round(np.mean(fair_loss_list[1]), 4)}, "
          f"Inactive Loss: {round(np.mean(fair_loss_list[2]), 4)}, "
          f"SDRO Weighted Loss: {round(np.mean(fair_loss_list[3]), 4)}")




@torch.no_grad()
def tune_new(epoch):

    overall_fair_result = evaluate_one_model(model_fair, tune_user_set,
                                             tune_item_set, tune_labels, tune_sample_num)
    active_fair_result = evaluate_one_model(model_fair, active_tune_user_set,
                                            active_tune_item_set, active_tune_labels, active_tune_sample_num)
    inactive_fair_result = evaluate_one_model(model_fair, inactive_tune_user_set,
                                              inactive_tune_item_set, inactive_tune_labels, inactive_tune_sample_num)


    ugf_fair = [abs(round(active_fair_result[i] - inactive_fair_result[i], 4))
                for i in range(len(active_fair_result))]
    if epoch > args.epochs // 2:
        save_model(model=model_fair, model_best_results=best_metric_result_fair,
                   this_ndcg_all=overall_fair_result[2], this_f1_all=overall_fair_result[3],
                   this_ndcg_active=active_fair_result[2], this_f1_active=active_fair_result[3],
                   this_ndcg_inactive=inactive_fair_result[2], this_f1_inactive=inactive_fair_result[3],
                   this_ndcg_ugf=ugf_fair[2], this_f1_ugf=ugf_fair[3], model_name="fair")

    print("tune:")
    print("\t\t\t\tLoss\tAcc\tNDCG\tf1")
    print(f"Fair\t\tOverall\t\t{str_format(overall_fair_result)}")
    print(f"\t\tActive\t\t{str_format(active_fair_result)}")
    print(f"\t\tInactive\t{str_format(inactive_fair_result)}")
    print(f"\t\tUGF\t\t{str_format(ugf_fair)}")


def save_model(model, model_best_results, this_ndcg_all, this_f1_all, this_ndcg_active, this_f1_active, this_ndcg_inactive,
               this_f1_inactive, this_ndcg_ugf, this_f1_ugf, model_name):
    if this_ndcg_all > model_best_results[BEST_NDCG_ALL]:
        model_best_results[BEST_NDCG_ALL] = this_ndcg_all
        torch.save(model.state_dict(), os.path.join(log, f"best_ndcg_all_{model_name}.pkl"))
    if this_f1_all > model_best_results[BEST_F1_ALL]:
        model_best_results[BEST_F1_ALL] = this_f1_all
        torch.save(model.state_dict(), os.path.join(log, f"best_f1_all_{model_name}.pkl"))
    if this_ndcg_active > model_best_results[BEST_NDCG_ACTIVE]:
        model_best_results[BEST_NDCG_ACTIVE] = this_ndcg_active
        torch.save(model.state_dict(), os.path.join(log, f"best_ndcg_active_{model_name}.pkl"))
    if this_f1_active > model_best_results[BEST_F1_ACTIVE]:
        model_best_results[BEST_F1_ACTIVE] = this_f1_active
        torch.save(model.state_dict(), os.path.join(log, f"best_f1_active_{model_name}.pkl"))
    if this_ndcg_inactive > model_best_results[BEST_NDCG_INACTIVE]:
        model_best_results[BEST_NDCG_INACTIVE] = this_ndcg_inactive
        torch.save(model.state_dict(), os.path.join(log, f"best_ndcg_inactive_{model_name}.pkl"))
    if this_f1_inactive > model_best_results[BEST_F1_INACTIVE]:
        model_best_results[BEST_F1_INACTIVE] = this_f1_inactive
        torch.save(model.state_dict(), os.path.join(log, f"best_f1_inactive_{model_name}.pkl"))
    if this_ndcg_ugf < model_best_results[BEST_NDCG_UGF]:
        model_best_results[BEST_NDCG_UGF] = this_ndcg_ugf
        torch.save(model.state_dict(), os.path.join(log, f"best_ndcg_ugf_{model_name}.pkl"))
    if this_f1_ugf < model_best_results[BEST_F1_UGF]:
        model_best_results[BEST_F1_UGF] = this_f1_ugf
        torch.save(model.state_dict(), os.path.join(log, f"best_f1_ugf_{model_name}.pkl"))


def str_format(data: list):
    return '\t'.join([f"{item:.4f}" for item in data])


@torch.no_grad()
def evaluate_one_model(model, user_set, item_set, labels, sample_num, if_test=False, result_num=0):
    predict = model(user_set, item_set)
    if if_test:
        results[result_num].append([int(user) for user in user_set])
        results[result_num].append([int(item) for item in item_set])
        results[result_num].append([float(score) for score in predict])
        results[result_num].append([float(label) for label in labels])
    begin = 0
    predict_numpy = []
    label_numpy = []
    predict_binary_numpy = []
    predict_label = [1 for _ in range(K)]
    temp_predict = predict.cpu()
    temp_label = labels.cpu()
    for num in sample_num:
        this_predict_label = temp_predict[begin:begin + num].view(-1)
        this_true_label = temp_label[begin:begin + num].view(-1)
        values, indices = torch.topk(this_predict_label, K)
        topk_predict = this_predict_label[indices]
        topk_label = this_true_label[indices]
        predict_numpy.append(topk_predict.numpy().reshape(1, -1))
        label_numpy.append(topk_label.cpu().numpy().reshape(1, -1))
        predict_binary_numpy.append(predict_label)
        begin += num
    label_numpy = np.array(label_numpy).squeeze()
    predict_numpy = np.array(predict_numpy).squeeze()
    predict_binary_numpy = np.array(predict_binary_numpy).squeeze()
    BCE_loss = round(bce_loss(predict.view(-1), labels).item(), 4)
    ndcg = round(ndcg_score(y_true=label_numpy, y_score=predict_numpy, k=K), 4)
    acc = round(accuracy_score(y_true=label_numpy.reshape(1, -1).squeeze(),
                               y_pred=predict_binary_numpy.reshape(1, -1).squeeze()), 4)
    f1 = round(f1_score(y_true=label_numpy.reshape(1, -1).squeeze(),
                        y_pred=predict_binary_numpy.reshape(1, -1).squeeze()), 4)
    # ndcg = round(ndcg_k(pred_label=predict, true_label=labels, k=K, sample_num=sample_num), 4)
    # f1 = round(precision_at_k(pred_label=predict, true_label=labels, k=K, sample_num=sample_num), 4)
    # hit_ratio = round(hitRatio_k(pred_label=predict, true_label=labels, k=K, sample_num=sample_num), 4)
    return [BCE_loss, acc, ndcg, f1]


@torch.no_grad()
def test_model():
    print(SPLIT)
    print("test!!!")
    metrics_name = ["best_ndcg_all", "best_f1_all", "best_ndcg_active", "best_f1_active", "best_ndcg_inactive", "best_f1_inactive", "best_ndcg_ugf", "best_f1_ugf"]
    for i in range(len(metrics_name)):
        metric = metrics_name[i]
        model_fair.load_state_dict(torch.load(os.path.join(log, f"{metric}_fair.pkl")))
        overall_fair_result = evaluate_one_model(model_fair, test_user_set,
                                                 test_item_set, test_labels, test_sample_num,if_test=True, result_num=i)
        active_fair_result = evaluate_one_model(model_fair, active_test_user_set,
                                                active_test_item_set, active_test_labels, active_test_sample_num)
        inactive_fair_result = evaluate_one_model(model_fair, inactive_test_user_set,
                                                  inactive_test_item_set, inactive_test_labels,
                                                  inactive_test_sample_num)

        ugf_fair = [abs(round(active_fair_result[i] - inactive_fair_result[i], 4))
                    for i in range(len(active_fair_result))]
        print(SPLIT)
        print(f"Result of {metric}")
        print("\t\t\t\tLoss\tAcc\tNDCG\tf1")
        print(f"Fair\t\tOverall\t\t{str_format(overall_fair_result)}")
        print(f"\t\tActive\t\t{str_format(active_fair_result)}")
        print(f"\t\tInactive\t{str_format(inactive_fair_result)}")
        print(f"\t\tUGF\t\t{str_format(ugf_fair)}")
        df = pd.DataFrame(
            data={"uid": results[i][RESULT_USER],
                  "iid": results[i][RESULT_ITEM],
                  "score": results[i][RESULT_SCORE],
                  "label": results[i][RESULT_LABEL]})
        filename = os.path.join(result_dir, f"{args.model}_{metric}_rank.csv")
        df.to_csv(filename, sep='\t', index=False)


model_dic_config = {
    'PMF': config.pmf_config,
    'VAECF': config.vaecf_config,
    'NeuMF': config.neumf_config,
    'NGCF': config.ngcf_config
}
args = parser.parse_args()
model = args.model
if args.model in model_dic_config:
    config = model_dic_config[args.model]
    print(f"Using {args.model} with configuration: {config}")
else:
    raise ValueError("Invalid model selected!")
set_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(f"cuda:{args.cuda_index}" if args.cuda else "cpu")
print(args)

log = os.path.join(args.log, '{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.epochs, args.neighbor_num, args.model_time))

if os.path.isdir(log):
    print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort." % log)
    time.sleep(5)
    os.system('rm -rf %s/' % log)

os.makedirs(log,exist_ok=True)
print("made the log directory", log)

result_dir = os.path.join("result", args.dataset)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
result_dir = os.path.join(result_dir, args.model)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
print(f"The result file will be stored in {result_dir}")
results = [[], [], [], [], [], [], [], []]

print(f"Train in {device}")
print(f"Dataset: {args.dataset}")
print(f"Train model: {args.model}")
print(f"L2: {L2}")

print(SPLIT)
print("Load dataset...")

dataset = myDatasetNew(dataset=args.dataset, train_neg_num=config["num_negative"],
                       neighbor_num=args.neighbor_num, result_path=result_dir)



(tune_user_set, tune_item_set, tune_labels, tune_sample_num), \
(test_user_set, test_item_set, test_labels, test_sample_num), \
(active_tune_user_set, active_tune_item_set, active_tune_labels, active_tune_sample_num), \
(active_test_user_set, active_test_item_set, active_test_labels, active_test_sample_num), \
(inactive_tune_user_set, inactive_tune_item_set, inactive_tune_labels, inactive_tune_sample_num), \
(inactive_test_user_set, inactive_test_item_set, inactive_test_labels, inactive_test_sample_num) = \
    dataset.instance_tune_test_set(choice="loo")
user_num, item_num, active_users, inactive_users = dataset.get_statistic()


active_indices = torch.tensor(active_users).to(device)
inactive_indices = torch.tensor(inactive_users).to(device)
tune_user_set = tune_user_set.to(device)
tune_item_set = tune_item_set.to(device)
tune_labels = tune_labels.to(device)
test_user_set = test_user_set.to(device)
test_item_set = test_item_set.to(device)
test_labels = test_labels.to(device)

active_tune_user_set = active_tune_user_set.to(device)
active_tune_item_set = active_tune_item_set.to(device)
active_tune_labels = active_tune_labels.to(device)
active_test_user_set = active_test_user_set.to(device)
active_test_item_set = active_test_item_set.to(device)
active_test_labels = active_test_labels.to(device)

inactive_tune_user_set = inactive_tune_user_set.to(device)
inactive_tune_item_set = inactive_tune_item_set.to(device)
inactive_tune_labels = inactive_tune_labels.to(device)
inactive_test_user_set = inactive_test_user_set.to(device)
inactive_test_item_set = inactive_test_item_set.to(device)
inactive_test_labels = inactive_test_labels.to(device)

print("Load succeed!")
print(SPLIT)

best_metric_result_original = [0, 0, 0, 0, 0, 0, 1e5, 1e5]
best_metric_result_fair = [0, 0, 0, 0, 0, 0, 1e5, 1e5]
#model:[PMF, VAECF, NeuMF, NGCF]
model_dic = {
    'PMF': PMF.PMF,
    'VAECF':VAECF.VAECF,
    'NeuMF': NeuMF.NeuMF,
    'NGCF':NGCF.NGCF
}
model_class = model_dic[args.model]

model_fair = model_class(config, user_num, item_num, device=device)
model_fair.to(device)
# model_list = [original, S-DRO, UFR, In-UCDS, In-Naive]



bce_loss = torch.nn.BCELoss()
my_loss = myLoss(l2=L2)

fair_model_opt = torch.optim.Adam(model_fair.parameters(),
                                  lr=config['adam_lr'],
                                  weight_decay=config['l2_regularization'])

sample_generator = dataset.instance_a_train_loader(config['batch_size'])

for epoch in range(args.epochs):
    train_SDRO(epoch)
    tune_new(epoch)
test_model()

