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
original_model_time = 0

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


#fair_model:original
def train_original(epoch):
    print(SPLIT)
    print(f"Train epoch: {epoch}")
    original_loss_list = [[], [], []]   #[total,bce,my_loss]
    for batch_idx, data in tqdm(enumerate(sample_generator)):
        users, items, rating = data[0].to(device), data[1].to(device), data[2].to(device)

        active_inactive_samples = dataset.get_most_similar_active_user(users)
        active_inactive_samples = active_inactive_samples.to(device)

        model_original.train()
        original_model_opt.zero_grad()
        # ----------------------------------------------------
        # train original model
        original_model_predict = model_original(users, items)

        original_model_bce_loss = bce_loss(original_model_predict.view(-1), rating)

        inactive_embedding_samples_original = model_original.inactive_embeddings_for_my_loss(active_inactive_samples)
        neighbor_embedding_samples_original = model_original.neighbor_embeddings_for_my_loss(active_inactive_samples)



        original_model_my_loss = my_loss(inactive_embedding_samples_original, neighbor_embedding_samples_original)

        original_model_loss = original_model_bce_loss
        original_loss_list[0].append(original_model_loss.item())
        original_loss_list[1].append(original_model_bce_loss.item())
        original_loss_list[2].append(original_model_my_loss.item())

        original_model_loss.backward()
        original_model_opt.step()


    print(f"Orin loss: {round(np.mean(original_loss_list[0]), 4)}, "
          f"BCE loss: {round(np.mean(original_loss_list[1]), 4)}, "
          f"my loss: {round(np.mean(original_loss_list[2]), 4)},")

    torch.cuda.empty_cache()



@torch.no_grad()
def tune_new(epoch):
    active_original_result = evaluate_one_model(model_original, active_tune_user_set,
                                                active_tune_item_set, active_tune_labels, active_tune_sample_num)
    inactive_original_result = evaluate_one_model(model_original, inactive_tune_user_set,
                                                  inactive_tune_item_set, inactive_tune_labels,
                                                  inactive_tune_sample_num)
    overall_original_result = evaluate_one_model(model_original, tune_user_set,
                                                 tune_item_set, tune_labels, tune_sample_num)


    ugf_original = [abs(round(active_original_result[i] - inactive_original_result[i], 4))
                    for i in range(len(active_original_result))]

    if epoch > args.epochs // 2:
        save_model(model=model_original, model_best_results=best_metric_result_original,
                   this_ndcg_all=overall_original_result[2], this_f1_all=overall_original_result[3],
                   this_ndcg_active=active_original_result[2], this_f1_active=active_original_result[3],
                   this_ndcg_inactive=inactive_original_result[2], this_f1_inactive=inactive_original_result[3],
                   this_ndcg_ugf=ugf_original[2], this_f1_ugf=ugf_original[3], model_name="original")

    print("tune:")
    print("\t\t\t\tLoss\tAcc\tNDCG\tf1")
    print(f"Ori\t\tOverall\t\t{str_format(overall_original_result)}")
    print(f"\t\tActive\t\t{str_format(active_original_result)}")
    print(f"\t\tInactive\t{str_format(inactive_original_result)}")
    print(f"\t\tUGF\t\t{str_format(ugf_original)}")


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
        model_original.load_state_dict(torch.load(os.path.join(log, f"{metric}_original.pkl")))
        active_original_result = evaluate_one_model(model_original, active_test_user_set,
                                                    active_test_item_set, active_test_labels, active_test_sample_num)
        inactive_original_result = evaluate_one_model(model_original, inactive_test_user_set,
                                                      inactive_test_item_set, inactive_test_labels,
                                                      inactive_test_sample_num)
        overall_original_result = evaluate_one_model(model_original, test_user_set,
                                                     test_item_set, test_labels, test_sample_num,
                                                     if_test=True, result_num=i)
        ugf_original = [abs(round(active_original_result[i] - inactive_original_result[i], 4))
                        for i in range(len(active_original_result))]
        print(SPLIT)
        print(f"Result of {metric}")
        print("\t\t\t\tLoss\tAcc\tNDCG\tf1")
        print(f"Ori\t\tOverall\t\t{str_format(overall_original_result)}")
        print(f"\t\tActive\t\t{str_format(active_original_result)}")
        print(f"\t\tInactive\t{str_format(inactive_original_result)}")
        print(f"\t\tUGF\t\t{str_format(ugf_original)}")
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

os.makedirs(log)
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
model_original = model_class(config, user_num, item_num, device=device)
model_original.to(device)
# method_list = [original, S-DRO, UFR, In-UCDS, In-Naive]

bce_loss = torch.nn.BCELoss()
my_loss = myLoss(l2=L2)


original_model_opt = torch.optim.Adam(model_original.parameters(),
                                      lr=config['adam_lr'],
                                      weight_decay=config['l2_regularization'])


sample_generator = dataset.instance_a_train_loader(config['batch_size'])

for epoch in range(args.epochs):
    train_original(epoch)
    tune_new(epoch)
test_model()
print(f"Original model run time: {original_model_time}")

