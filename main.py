import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
import config as cfg
from model import deepfm
from model.deepfm import gini_norm
# from memory_profiler import profile
import gc

logger = cfg.logger


# # 由于基尼系数越大越好，所以用到了make_scorer   greater_is_better=true时越大越好
# gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)

# @profile()
def load_data():
    train_df = pd.read_csv(cfg.TRAIN_FILE)
    test_df = pd.read_csv(cfg.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ['id', 'target']]
        # df['missing_feat'] = np.sum(df[df[cols]==-1].values,axis=1)
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        return df

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    # x_train 需要排除 id，target列，y是target列
    cols = [i for i in train_df.columns if i not in ['id', 'target']]
    x_train = train_df[cols].values
    y_train = train_df['target'].values
    # 测试数据集没有target列，所以需要置为-1
    x_test = test_df[cols].values
    # y_test = test_df['target'].values
    # y_test = -1

    # 返回带列名的x，不带列名的x，和Y
    return train_df, test_df, x_train, x_test, y_train

    del train_df
    gc.collect()


def emb_feat(train_df, test_df, numeuic_cols, ignore_cols):
    """
    :param train_df:
    :param test_df:
    :param numeuic_cols:
    :param ignore_cols:
    :return:feat_dict(特征索引), total_count（所有的特征数量，含不同的离散变量）
    """
    feat_dict = {}
    total_count = 0
    df = pd.concat([train_df, test_df])
    for col in df.columns:
        if col in ignore_cols:
            continue
        elif col in numeuic_cols:
            feat_dict[col] = total_count
            total_count += 1
        else:
            us = df[col].unique()
            # 这个里面对应的仍然是一个字典 里面的字典  key为每一个唯一的离散的特征，value为唯一索引（自增）
            feat_dict[col] = dict(zip(us, range(total_count, total_count + len(us))))
            total_count += len(us)
    return feat_dict, total_count


def feat_parse(feat_dict, df, has_label=False):
    df_i = df.copy()
    if has_label:
        y = df_i['target'].values.tolist()
        df_i.drop(['id', 'target'], axis=1, inplace=True)
    else:  # 测试数据集没有target
        ids = df_i['id'].values.tolist()
        df_i.drop(['id'], axis=1, inplace=True)
    # df_i for feature index
    # df_v for featur value which can be either binary(1/0) or float(e.g., 0.8899)
    df_v = df_i.copy()

    for col in df_i.columns:
        if col in cfg.IGNORE_COLS:
            df_i.drop(col, axis=1, inplace=True)
            df_v.drop(col, axis=1, inplace=True)
            continue
        elif col in cfg.NUMERIC_COLS:
            df_i[col] = feat_dict[col]
            # 这个少一个df_v啊  因为dv_v本来就是copy的train_df 所以不需要做任何更改
        # 离散变量
        else:
            # 取到字典中对应的索引值
            df_i[col] = df_i[col].map(feat_dict[col])
            df_v[col] = 1
        x_i = df_i.values.tolist()
        x_v = df_v.values.tolist()
        if has_label:
            return x_i, x_v, y
        else:
            return x_i, x_v, ids


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(cfg.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1] + 1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], linestyle="dashed", marker="o")
        legends.append("train-%d" % (i + 1))
        legends.append("valid-%d" % (i + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("fig/%s.png" % model_name)
    plt.close()


if __name__ == '__main__':
    train_df, test_df, x_train, x_test, y_train = load_data()

    # 交叉验证,[(x1,y1),(x2,y2).....]
    folds = list(
        StratifiedKFold(n_splits=cfg.NUM_SPLITS, shuffle=True, random_state=cfg.RANDOM_SEED).split(x_train, y_train))
    # 处理特征 变成embedding
    feat_dict, total_count = emb_feat(train_df, test_df, cfg.NUMERIC_COLS, cfg.IGNORE_COLS)
    xi_train, xv_train, y_train = feat_parse(feat_dict=feat_dict, df=train_df, has_label=True)
    xi_test, xv_test, ids_test = feat_parse(feat_dict=feat_dict, df=test_df)

    dfm_params = cfg.dfm_params

    # 包含离散特征的取值  一共有多少个 例如年龄分别有12,14,15  total_count=3
    dfm_params['feature_size'] = total_count
    # 感觉像样本数量   也可能是特征数 例如年龄，性别  这就是2个特征
    dfm_params['field_size'] = len(xi_train[0])

    _get = lambda x, l: [x[i] for i in l]  # 假如l传入的是df，那么i即是列名

    # len(folds) 应该是folds分了多少层 就是多少，不是就应该等于cfg.NUM_SPLITS？  这个不懂 是不是4*30
    y_train_meta = np.zeros((train_df.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((test_df.shape[0], 1), dtype=float)
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params['epoch']), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params['epoch']), dtype=float)

    # folds 返回的是行索引   所以_get 匿名函数即是取kfold中分出来的数据集而已
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(xi_train, train_idx), _get(xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(xi_train, valid_idx), _get(xv_train, valid_idx), _get(y_train, valid_idx)

        # 就是每一批训练了   重中之重-------------------------------------------------------------------------
        dfm = deepfm.DeepFm(**dfm_params)

        # fit
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:, 0] += dfm.predict(xi_test, xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))
    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)" % (clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv" % (clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    print(y_train_meta, y_test_meta)  # 其实是验证数据集和测试数据集
