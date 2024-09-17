import json

from collections import defaultdict


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn import metrics
from multiprocessing import Pool
from matplotlib import pyplot as plt
import sys
from file_reader import load_protein, load_feature, load_annotation
from ontology import HumanPhenotypeOntology, get_ns_id
import os
import optuna
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = os.getcwd()
# use_gpu = torch.cuda.is_available()

class ThreeLayerNN(nn.Module):
    def __init__(self, data_in=19029, hidden1_node=100, data_out=9010, p=0.5):
        global feature_size, annotation_size
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden1_node, bias=True)
        self.fc2 = nn.Linear(hidden1_node, annotation_size, bias=True)
        self.sg = nn.Sigmoid()
        self.dropout = nn.Dropout(p=p)
        self.loss_func = nn.BCELoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sg(x)
        return x

def seed_torch(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # if use_gpu:
    #     torch.cuda.manual_seed(seed)
    # else:
    #     torch.manual_seed(seed)
    torch.manual_seed(seed)

def auroc(result, annotation):
    """Calculate term-centric AUROC
    :param result: predictive scores, DataFrame like
        { protein1: { hpo_term1: score1, ... }, ... }
    :param annotation: true HPO annotations, DataFrame like
        { protein1: { hpo_term1: 0/1, ... }, ... } (0: no, 1: yes)
    :return: term-centric AUROC
    """
    auc = 0
    n_terms = 0

    # scan each HPO term
    for i in range(annotation.shape[1]):
        y_true = annotation[:, i].tolist()
        y_pred = result[:, i].tolist()

        # if only one class
        if len(np.unique(y_true)) != 2:
            continue

        n_terms += 1
        auc += metrics.roc_auc_score(y_true, y_pred)
    # average over terms
    auc = auc / n_terms if n_terms > 0 else 0

    return auc

def f_run(result, annotation, threshold):
    n_proteins, n_proteins_covered = 0, 0
    precision_ths, recall_ths = 0, 0

    # calculate precision and recall of each protein
    for i in range(annotation.shape[0]):
        correct = 0
        # HPO terms retrieved above the given threshold
        retrieved = len(result[i][result[i] >= threshold])
        # HPO terms associated with the given protein
        relevant = len(annotation[i][annotation[i] == 1])
        for j in range(annotation.shape[1]):
            correct += result[i][j] >= threshold and annotation[i][j] == 1

        if relevant > 0:
            # total of proteins having HPO annotations
            n_proteins += 1
            if retrieved > 0:
                # total of proteins annotated at least on predicted HPO term
                n_proteins_covered += 1
                precision_ths += correct / retrieved
            recall_ths += correct / relevant

    try:
        precision_ths = precision_ths / n_proteins_covered
    except ZeroDivisionError:
        precision_ths = 0
    try:
        recall_ths /= n_proteins
    except ZeroDivisionError:
        recall_ths = 0
    try:
        f_max_ths = (2 * precision_ths * recall_ths) / (precision_ths + recall_ths)
    except ZeroDivisionError:
        f_max_ths = 0
    return f_max_ths, threshold

def multiProcessing_f_max(result, annotation, n_threshold=101):
    f_max_applyResult = []
    f_max_list = list()
    threshold_list = []
    pool = Pool()
    for threshold in np.linspace(0., 1., n_threshold):
        f_max_applyResult.append(pool.apply_async(f_run, (result, annotation, threshold)))
    pool.close()
    pool.join()

    for result in f_max_applyResult:
        f_max_list.append(result.get()[0])
        threshold_list.append(result.get()[1])
    f_max_overall = max(f_max_list)
    # get the corresponding threshold
    threshold = threshold_list[np.argmax(f_max_list)]

    return f_max_overall, threshold

def f_max(result, annotation, n_threshold=101):
    """Calculate F-max and the corresponding threshold
    :param result: predictive scores, DataFrame like
        { protein1: { hpo_term1: score1, ... }, ... }
    :param annotation: true HPO annotations, DataFrame like
        { protein1: { hpo_term1: 0/1, ... }, ... } (0: no, 1: yes)
    :param n_threshold: number of thresholds, default: 101 (i.e. step=0.01)
    :return: F-max and the corresponding threshold
    """
    f_max_list = list()

    # vary the threshold over the range between 0 to 1 with n_threshold steps
    for threshold in np.linspace(0., 1., n_threshold):
        n_proteins, n_proteins_covered = 0, 0
        precision_ths, recall_ths = 0, 0

        # calculate precision and recall of each protein
        for i in range(annotation.shape[0]):
            correct = 0
            # HPO terms retrieved above the given threshold
            retrieved = len(result[i][result[i] >= threshold])
            # HPO terms associated with the given protein
            relevant = len(annotation[i][annotation[i] == 1])
            for j in range(annotation.shape[1]):
                correct += result[i][j] >= threshold and annotation[i][j] == 1

            if relevant > 0:
                # total of proteins having HPO annotations
                n_proteins += 1
                if retrieved > 0:
                    # total of proteins annotated at least on predicted HPO term
                    n_proteins_covered += 1
                    precision_ths += correct / retrieved
                recall_ths += correct / relevant

        try:
            precision_ths = precision_ths / n_proteins_covered
        except ZeroDivisionError:
            precision_ths = 0
        try:
            recall_ths /= n_proteins
        except ZeroDivisionError:
            recall_ths = 0
        try:
            f_max_ths = (2*precision_ths*recall_ths)/(precision_ths+recall_ths)
        except ZeroDivisionError:
            f_max_ths = 0
        f_max_list.append(f_max_ths)

    # search the highest F-max value
    f_max_overall = max(f_max_list)
    # get the corresponding threshold
    threshold = np.argmax(f_max_list) / (n_threshold - 1)

    return f_max_overall, threshold

def aupr(result, annotation):
    """Calculate pairwise AUPR
    :param result: predictive scores, DataFrame like
        { protein1: { hpo_term1: score1, ... }, ... }
    :param annotation: true HPO annotations, DataFrame like
        { protein1: { hpo_term1: 0/1, ... }, ... } (0: no, 1: yes)
    :return: pairwise AUPR
    """
    y_true = annotation.reshape(-1)
    y_pred = result.reshape(-1)
    aupr_value = metrics.average_precision_score(y_true, y_pred)
    return aupr_value

# def train(model, X_train, y_train, X_val, y_val, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):
#     train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
#     # val_loader = DataLoader(TensorDataset(X_val, y_val), BATCH_SIZE, shuffle=True)
#
#     loss_func = model.loss_func
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     if(use_gpu):
#         loss_func = loss_func.cuda()
#
#     losses = []
#     val_losses = []
#     train_acc = []
#     val_acc = []
#     eval = []
#     count = 0
#
#     for epoch in range(TOTAL_EPOCHS):
#         count += 1
#         model.train()
#         correct = 0  # 记录正确的个数，每个epoch训练完成之后打印accuracy
#         for i, (features, labels) in enumerate(train_loader):
#             features = features.float()
#             labels = torch.squeeze(labels.type(torch.float))
#             optimizer.zero_grad()  # 清零
#             outputs = model(features)
#             # 计算损失函数
#             loss = loss_func(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             # 计算正确率
#             # correct += auroc(outputs, labels) * len(labels)
#
#             if (i + 1) % 10 == 0:
#                 # 每10个batches打印一次loss
#                 print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
#                                                                     i + 1, len(X_train) // BATCH_SIZE,
#                                                                     loss.item()))
#         losses.append(loss.item())
#         accuracy = 100. * correct / len(X_train)
#         print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(
#             epoch + 1, loss.item(), correct, len(X_train), accuracy))
#         train_acc.append(accuracy)
#
#         # 每个epoch计算测试集accuracy
#         model.eval()
#         val_loss = 0
#         accuracy = 0
#         with torch.no_grad():
#             X_val = X_val.float()
#             y_val = y_val.float()
#             optimizer.zero_grad()
#             pred_y = model(X_val)
#             val_loss = loss_func(pred_y, y_val).item()
#             # accuracy = auroc(pred_y, y_val)
#             if count == TOTAL_EPOCHS:
#                 fmax, threshold = multiProcessing_f_max(pred_y.cpu().detach().numpy(), y_val.cpu().detach().numpy())
#                 accuracy = auroc(pred_y, y_val)
#                 aupr_val = aupr(pred_y.cpu().detach().numpy(), y_val.cpu().detach().numpy())
#                 eval.append(fmax)
#                 eval.append(threshold)
#                 eval.append(accuracy)
#                 eval.append(aupr_val)
#
#
#         val_losses.append(val_loss)
#         print('Test set: Average loss: {:.4f}, Accuracy: ({:.3f}%)\n'.format(
#             val_loss, accuracy))
#         val_acc.append(accuracy)
#
#     torch.save(model, path_PPI_model)
#
#     return losses, val_losses, train_acc, val_acc, eval

def train(model, X_train, y_train, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):
    train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    loss_func = model.loss_func
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    count = 0

    for epoch in range(TOTAL_EPOCHS):
        count += 1
        model.train()
        correct = 0  # 记录正确的个数，每个epoch训练完成之后打印accuracy
        for i, (features, labels) in enumerate(train_loader):

            features = features.float()

            optimizer.zero_grad()  # 清零
            outputs = model(features)
            # 计算损失函数

            loss = loss_func(outputs, labels.type(torch.float))
            loss.backward()
            optimizer.step()
            # 计算正确率

            if (i + 1) % 10 == 0:
                # 每10个batches打印一次loss
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    loss.item()))
        accuracy = 100. * correct / len(X_train)
        print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(
            epoch + 1, loss.item(), correct, len(X_train), accuracy))

def prediction(model, X_val):
    '''
    predic HPO term annotation as model
    :param model:
    :param X_val: Tensor
    :return:
    '''
    model.eval()
    torch.no_grad
    with torch.no_grad():
        pred = model(X_val)
    return pred

def get_kfold_data(k, i, X, y):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim=0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim=0)
    else:  # 若是最后一折交叉验证
        X_valid, y_valid = X[val_start:], y[val_start:]  # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]
        y_train = y[0:val_start]

    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs=3, learning_rate=0.0001, batch_size=16):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    valid_fmax, valid_acc, valid_aupr = 0, 0, 0

    for i in range(k):
        print('*' * 25, '第', i + 1, '折', '*' * 25)
        data = get_kfold_data(k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据
        model = ThreeLayerNN()  # 实例化模型（某已经定义好的模型）
        # if(use_gpu):
        #     model = model.cuda()
        # 每份数据进行训练
        train_loss, val_loss, train_acc, val_acc, eval = train(model, *data, batch_size, learning_rate, num_epochs)

        plt.plot(train_loss, color='orange')
        plt.plot(val_loss, color='pink')
        plt.savefig("All_loss" + str(num_epochs) + ".png")

        print('train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss[-1], train_acc[-1]))
        print('valid loss:{:.5f}, valid_acc:{:.3f}%\n'.format(val_loss[-1], val_acc[-1]))

        print('valid fmax:{:.5f}, threshold:{:.2f}, valid acc:{:.5f}, valid aupr:{:.5f}'.format(eval[0], eval[1], eval[2], eval[3]))

        train_loss_sum += train_loss[-1]
        valid_loss_sum += val_loss[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += val_acc[-1]

        valid_fmax += eval[0]
        valid_acc += eval[2]
        valid_aupr += eval[3]

    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
    print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))

    print('average valid fmax:{:.5f}'.format(valid_fmax / k))
    print('average valid acc:{:.5f}'.format(valid_acc / k))
    print('average valid aupr:{:.5f}'.format(valid_aupr / k))

    return valid_fmax / k, valid_acc / k, valid_aupr / k

# estimator = dp.PCA(n_components=0.99, random_state=1)

# PPI_feature = pd.read_csv(path_PPI_feature, index_col=0, low_memory=False).fillna(0).values
# # PPI_feature = estimator.fit_transform(PPI_feature)
# PPI_label = pd.read_csv(path_PPI_label, index_col=0, low_memory=False).values
# train_x = torch.from_numpy(PPI_feature)
# train_y = torch.from_numpy(PPI_label)
# feature_size = train_x.shape[1]
# learning_rate = 1e-4
# epochs = 10
# batch_size = 10
#
# if(use_gpu):
#     train_x = train_x.cuda()
#     train_y = train_y.cuda()
#
# opts, args = getopt(sys.argv[1:], "l:b:e:", ['learning_rate=', 'batch_size=', 'epochs='])
# for opt, arg in opts:
#     if opt in ('-l', '--learning_rate'):
#         learning_rate = float(arg)
#     elif opt in ('-b', '--batch_size'):
#         batch_size = int(arg)
#     elif opt in ('-e', '--epochs'):
#         epochs = int(arg)
#
# fmax, auc, aupr = k_fold(10, train_x, train_y, epochs, learning_rate, batch_size)
#
# raw = [learning_rate, batch_size, epochs, fmax, auc, aupr]
# with open("../data/clean/get_parameter_PPI.txt", 'a') as fp:
#     fp.writelines(raw)


def objective(trial):
    seed_torch()

    hidden_nodes = trial.suggest_int('hidden_nodes', 1500, 2500)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    batch_size = trial.suggest_int('batch_size', 4, 64)
    epochs = 10  # 固定训练轮数

    model = ThreeLayerNN(data_in=feature_size, hidden1_node=hidden_nodes, data_out=annotation_size, p=dropout)
    # if use_gpu:
    #     model.cuda()

    # train_optuna_x = torch.from_numpy(train_optuna_feature).float()
    # train_optuna_y = torch.from_numpy(train_optuna_annotation).float()
    # val_optuna_x = torch.from_numpy(val_optuna_feature).float()
    # val_optuna_y = torch.from_numpy(val_optuna_annotation).float()
    #
    # train(model, train_optuna_x, train_optuna_y, batch_size, learning_rate, epochs)
    # val_optuna_pred = prediction(model, val_optuna_x)
    # val_optuna_loss = model.loss_func(val_optuna_pred, val_optuna_y.type(torch.float))#.item()
    #
    # if np.isnan(val_optuna_loss):
    #     val_optuna_loss =np.inf
    #
    # return val_optuna_loss


    train(model, train_x, train_y, batch_size, learning_rate, epochs)
    ltr_pred = prediction(model, ltr_x)
    ltr_loss = model.loss_func(ltr_pred, ltr_y.type(torch.float))#.item()

    if np.isnan(ltr_loss):
        ltr_loss =np.inf

    return ltr_loss


if __name__ == "__main__":
    seed_torch()
    conf_path = "../../../config/basic/dnn/dnn_TF-IDF_all_full.json"
    if len(sys.argv) > 1 and len(sys.argv[1]) > 0:
        conf_path = sys.argv[1]

    with open(conf_path) as fp:
        config = json.load(fp)

    # load HPO
    ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                      version=config["ontology"]["version"])

    # load training set, ltr training set and test set
    train_protein_list = load_protein(config["protein_list"]["train"])
    ltr_protein_list = load_protein(config["protein_list"]["ltr"])
    test_protein_list = load_protein(config["protein_list"]["test"])
    hpoterm_all_list=load_protein(config["term_list"])

    # load features and convert them to DataFrame
    feature = load_feature(config["feature"])
    df_feature = pd.DataFrame.from_dict(feature, orient="index")
    df_feature = df_feature.fillna(0)

    combined_ltr_result = defaultdict(dict)
    combined_test_result = defaultdict(dict)
#####


    # load propagated HPO annotations of specified sub-ontology
    hpo_annotation = load_annotation(config["annotation"], ontology, ns='all')
    # transform it to double-layer dict, i.e.
    # { protein1: { hpo_term1: score1, hpo_term2: score2, ... }, ... }
    dict_annotation = defaultdict(dict)
    for protein in hpo_annotation:
        for hpo_term in hpo_annotation[protein]:
            dict_annotation[protein][hpo_term] = 1
    # convert annotation to DataFrame
    df_annotation = pd.DataFrame.from_dict(dict_annotation, orient="index",columns=list(hpoterm_all_list),dtype=object)
    df_annotation = df_annotation.fillna(0)

    # extract training features and annotations
    train_protein_of_ns = list(set(train_protein_list) &
                               set(df_annotation.axes[0]) &
                               set(df_feature.axes[0]))

    train_feature = df_feature.loc[train_protein_of_ns].values
    train_annotation = df_annotation.loc[train_protein_of_ns]
    columns = list(train_annotation.columns)
    train_annotation = train_annotation.values



    # # extract ltr training features and annotations

    #####
    # load propagated HPO annotations of specified sub-ontology
    hpo_ltr_annotation = load_annotation(config["ltr_annotation"], ontology, ns='all')
    # transform it to double-layer dict, i.e.
    # { protein1: { hpo_term1: score1, hpo_term2: score2, ... }, ... }
    dict_ltr_annotation = defaultdict(dict)
    for protein in hpo_ltr_annotation:
        for hpo_term in hpo_ltr_annotation[protein]:
            dict_ltr_annotation[protein][hpo_term] = 1
    # convert annotation to DataFrame
    df_ltr_annotation = pd.DataFrame.from_dict(dict_ltr_annotation, orient="index",columns=list(hpoterm_all_list),dtype=object)
    df_ltr_annotation = df_ltr_annotation.fillna(0)

    # extract training features and annotations
    ltr_protein_of_ns = list(set(ltr_protein_list) &
                               set(df_ltr_annotation.axes[0]) &
                               set(df_feature.axes[0]))
    ltr_feature = df_feature.loc[ltr_protein_of_ns]#.values
    ltr_annotation = df_ltr_annotation.loc[ltr_protein_of_ns]

    ltr_annotation = ltr_annotation.values
    #
    # ltr_protein_of_ns = list(set(ltr_protein_list) &
    #                          set(df_feature.axes[0]))


    # print("ltr_protein_of_ns",ltr_protein_of_ns)

    # ltr_feature = df_feature.loc[ltr_protein_of_ns]
    ltr_index = list(ltr_feature.index)
    ltr_feature = ltr_feature.values


    # extract test features and annotations
    test_protein_of_ns = list(set(test_protein_list) &
                              set(df_feature.axes[0]))
    test_feature = df_feature.loc[test_protein_of_ns]
    test_index = list(test_feature.index)
    test_feature = test_feature.values

    # ltr_annotation = df_ltr_annotation.loc[ltr_protein_of_ns]
    # ltr_annotation = ltr_annotation.values


    # from sklearn.model_selection import train_test_split
    #
    #
    # train_optuna_feature, val_optuna_feature, train_optuna_annotation, val_optuna_annotation = train_test_split(
    #     train_feature, train_annotation, test_size=0.1, random_state=42
    # )
    #
    # # 将 numpy 数组转换为 PyTorch 张量
    # train_optuna_x = torch.from_numpy(train_optuna_feature).float()
    # train_optuna_y = torch.from_numpy(train_optuna_annotation).float()
    # val_optuna_x = torch.from_numpy(val_optuna_feature).float()
    # val_optuna_y = torch.from_numpy(val_optuna_annotation).float()



    train_x = torch.from_numpy(train_feature)
    train_y = torch.from_numpy(train_annotation)
    ltr_x = torch.from_numpy(ltr_feature).float()
    ltr_y = torch.from_numpy(ltr_annotation)
    test_x = torch.from_numpy(test_feature).float()

    feature_size = train_x.shape[1]
    annotation_size = train_y.shape[1]
    # learning_rate = config["learning_rate"]
    # epochs = config["epochs"]
    # batch_size = config["batch_size"]
    # hidden_nodes = config["hidden_nodes"]
    # dropout = config["dropout"]
    #
    # # train model and predict
    # model = ThreeLayerNN(hidden1_node=hidden_nodes, p=dropout)  # 实例化模型（某已经定义好的模型）
    #
    #
    #
    #
    #
    # train(model, train_x, train_y, batch_size, learning_rate, epochs)
    # ltr_result = prediction(model, ltr_x)
    # ltr_result = pd.DataFrame(ltr_result.cpu().numpy(), index=ltr_index, columns=columns)
    # ltr_result = ltr_result.to_json(config["result"]["ltr"], orient='index')# indent=2, orient='index')
    #
    # test_result = prediction(model, test_x)
    # test_result = pd.DataFrame(test_result.cpu().numpy(), index=test_index, columns=columns)
    # test_result = test_result.to_json(config["result"]["test"], orient='index')# indent=2, orient='index')


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print(f"Best trial: {study.best_trial.value}")
    print("Best hyperparameters: ", study.best_trial.params)

    # 使用最佳超参数训练最终模型
    best_params = study.best_trial.params
    final_model = ThreeLayerNN(data_in=feature_size, hidden1_node=best_params['hidden_nodes'], data_out=annotation_size,
                               p=best_params['dropout'])
    # if use_gpu:
    #     final_model.cuda()

    train(final_model, train_x, train_y, best_params['batch_size'], best_params['learning_rate'], 20)
    ltr_result = prediction(final_model, ltr_x)
    ltr_result = pd.DataFrame(ltr_result.cpu().numpy(), index=ltr_index, columns=list(df_annotation.columns))
    ltr_result.to_json(config["result"]["ltr"], orient='index')

    test_result = prediction(final_model, test_x)
    test_result = pd.DataFrame(test_result.cpu().numpy(), index=test_index, columns=list(df_annotation.columns))
    test_result.to_json(config["result"]["test"], orient='index')


    # ns_id = get_ns_id(version=config["ontology"]["version"])
    # for ns in ns_id:
    #     # load propagated HPO annotations of specified sub-ontology
    #     hpo_annotation = load_annotation(config["annotation"], ontology, ns)
    #     if len(hpo_annotation) == 0:
    #         continue
    #     # transform it to double-layer dict, i.e.
    #     # { protein1: { hpo_term1: score1, hpo_term2: score2, ... }, ... }
    #     dict_annotation = defaultdict(dict)
    #     for protein in hpo_annotation:
    #         for hpo_term in hpo_annotation[protein]:
    #             dict_annotation[protein][hpo_term] = 1
    #     # convert annotation to DataFrame
    #     df_annotation = pd.DataFrame.from_dict(dict_annotation, orient="index")
    #     df_annotation = df_annotation.fillna(0)
    #
    #     # extract training features and annotations
    #     train_protein_of_ns = list(set(train_protein_list) &
    #                                set(df_annotation.axes[0]) &
    #                                set(df_feature.axes[0]))
    #     train_feature = df_feature.loc[train_protein_of_ns].values
    #     train_annotation = df_annotation.loc[train_protein_of_ns]
    #     columns = list(train_annotation.columns)
    #     train_annotation = train_annotation.values
    #     # extract ltr training features and annotations
    #     ltr_protein_of_ns = list(set(ltr_protein_list) &
    #                              set(df_feature.axes[0]))
    #     ltr_feature = df_feature.loc[ltr_protein_of_ns]
    #     ltr_index = list(ltr_feature.index)
    #     ltr_feature = ltr_feature.values
    #     # extract test features and annotations
    #     test_protein_of_ns = list(set(test_protein_list) &
    #                               set(df_feature.axes[0]))
    #     test_feature = df_feature.loc[test_protein_of_ns]
    #     test_index = list(test_feature.index)
    #     test_feature = test_feature.values
    #
    #     train_x = torch.from_numpy(train_feature)
    #     train_y = torch.from_numpy(train_annotation)
    #     ltr_x = torch.from_numpy(ltr_feature).float()
    #     test_x = torch.from_numpy(test_feature).float()
    #     feature_size = train_x.shape[1]
    #     annotation_size = train_y.shape[1]
    #     learning_rate = config["learning_rate"]
    #     epochs = config["epochs"]
    #     batch_size = config["batch_size"]
    #     hidden_nodes = config["hidden_nodes"]
    #     dropout = config["dropout"]
    #
    #     # train model and predict
    #     model = ThreeLayerNN(hidden1_node=hidden_nodes, p=dropout)  # 实例化模型（某已经定义好的模型）
    #     # if (use_gpu):
    #     #     model = model.cuda()
    #     #     train_x = train_x.cuda()
    #     #     train_y = train_y.cuda()
    #     #     ltr_x = ltr_x.cuda()
    #     #     test_x = test_x.cuda()
    #     train(model, train_x, train_y, batch_size, learning_rate, epochs)
    #     ltr_result = prediction(model, ltr_x)
    #     ltr_result = pd.DataFrame(ltr_result.cpu().numpy(), index=ltr_index, columns=columns)
    #     test_result = prediction(model, test_x)
    #     test_result = pd.DataFrame(test_result.cpu().numpy(), index=test_index, columns=columns)
    #
    #     ltr_result = ltr_result.to_json("ltr_dnn_d2v_temp.json", indent=2, orient='index')
    #     test_result = test_result.to_json("test_dnn_d2v_temp.json", indent=2, orient='index')
    #
    #     # combine result into final result
    #     with open("ltr_dnn_d2v_temp.json") as fp:
    #         ltr_temp= json.load(fp)
    #     with open("test_dnn_d2v_temp.json") as fp:
    #         test_temp= json.load(fp)
    #     for protein in ltr_temp:
    #         for term in ltr_temp[protein]:
    #             combined_ltr_result[protein][term] = ltr_temp[protein][term]
    #     for protein in test_temp:
    #         for term in test_temp[protein]:
    #             combined_test_result[protein][term] = test_temp[protein][term]
    #     os.remove("ltr_dnn_d2v_temp.json")
    #     os.remove("test_dnn_d2v_temp.json")
    #
    # # write result
    # with open(config["result"]["ltr"], "w") as fp:
    #     json.dump(combined_ltr_result, fp, indent=2)
    # with open(config["result"]["test"], "w") as fp:
    #     json.dump(combined_test_result, fp, indent=2)
