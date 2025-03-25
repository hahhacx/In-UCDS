import argparse
import numpy as np
import torch
import os
import config
from sklearn.metrics import ndcg_score, f1_score, accuracy_score
from sigdatasets import myDatasetNew
import random
import models.PMF as PMF
import models.NeuMF as NeuMF
import models.NGCF as NGCF
import models.VAECF as VAECF

SPLIT = "=" * 60
K = 10
L2 = 1e-5
original_model_time = 0
fair_model_time = 0

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


def str_format(data: list):
    return '\t'.join([f"{item:.4f}" for item in data])


@torch.no_grad()
def evaluate_one_model(model, user_set, item_set, labels, sample_num):
    predict = model(user_set, item_set)
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
        model_original.load_state_dict(torch.load(os.path.join(log, f"{metric}_original.pkl")))
        model_fair.load_state_dict(torch.load(os.path.join(log, f"{metric}_fair.pkl")))
        active_original_result = evaluate_one_model(model_original, active_test_user_set,
                                                    active_test_item_set, active_test_labels, active_test_sample_num)
        inactive_original_result = evaluate_one_model(model_original, inactive_test_user_set,
                                                      inactive_test_item_set, inactive_test_labels,
                                                      inactive_test_sample_num)
        overall_original_result = evaluate_one_model(model_original, test_user_set,
                                                     test_item_set, test_labels, test_sample_num)
        overall_fair_result = evaluate_one_model(model_fair, test_user_set,
                                                 test_item_set, test_labels, test_sample_num)
        active_fair_result = evaluate_one_model(model_fair, active_test_user_set,
                                                active_test_item_set, active_test_labels, active_test_sample_num)
        inactive_fair_result = evaluate_one_model(model_fair, inactive_test_user_set,
                                                  inactive_test_item_set, inactive_test_labels,
                                                  inactive_test_sample_num)
        ugf_original = [abs(round(active_original_result[i] - inactive_original_result[i], 4))
                        for i in range(len(active_original_result))]

        ugf_fair = [abs(round(active_fair_result[i] - inactive_fair_result[i], 4))
                    for i in range(len(active_fair_result))]
        print(SPLIT)
        print(f"Result of {metric}")
        print("\t\t\t\tLoss\tAcc\tNDCG\tf1")
        print(f"Ori\t\tOverall\t\t{str_format(overall_original_result)}")
        print(f"\t\tActive\t\t{str_format(active_original_result)}")
        print(f"\t\tInactive\t{str_format(inactive_original_result)}")
        print(f"\t\tUGF\t\t{str_format(ugf_original)}")
        print(f"Fair\t\tOverall\t\t{str_format(overall_fair_result)}")
        print(f"\t\tActive\t\t{str_format(active_fair_result)}")
        print(f"\t\tInactive\t{str_format(inactive_fair_result)}")
        print(f"\t\tUGF\t\t{str_format(ugf_fair)}")


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

if not os.path.exists(log):
    raise FileNotFoundError(f"Error: The log path '{log}' does not exist.")
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
print(f"Train method: {args.model}")
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

#model:[PMF, VAECF, NeuMF, NGCF]
model_dic = {
    'PMF': PMF.PMF,
    'VAECF':VAECF.VAECF,
    'NeuMF': NeuMF.NeuMF,
    'NGCF':NGCF.NGCF
}
model_class = model_dic[args.model]
model_original = model_class(config, user_num, item_num, device=device)
model_fair = model_class(config, user_num, item_num, device=device)
model_original.to(device)
model_fair.to(device)
# method_list = [original, S-DRO, UFR, In-UCDS, In-Naive]


bce_loss = torch.nn.BCELoss()



test_model()

