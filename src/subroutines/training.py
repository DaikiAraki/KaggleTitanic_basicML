# Copyright 2022 Daiki Araki. All Rights Reserved.

import random
import dill
import numpy
import numpy as np

from src.settings.Config import Config
from src.shared_functions.io_dataset import dataset_read_titanic_train
from src.shared_functions.titanic_data_conversion import input_column_names
from src.shared_functions.plot_performances import plot_performances
from src.models.KNearestNeighborsHandler import KNearestNeighborsHandler
from src.models.LogisticRegressionHandler import LogisticRegressionHandler
from src.models.SVCHandler import SVCHandler

"""
K-Nearest Neighbors, Logistic Regression, SVCのモデルを最適化して保存する。基本的に結果はファイルに出力される。
保存先: "(srcフォルダの親dir)\\data"
"""

def training(cfg):

    if cfg is None:
        cfg = Config()

    # titanic.zipからデータ読み出し
    X, T_class, T_value = dataset_read_titanic_train(path=cfg.path_titanic_zip, zip_mode=True, name="train.csv")

    # np.set_printoptions(threshold=10000000)
    # print(str(np.any(np.isnan(X))))
    # print(str(np.any(np.isnan(T_class))))
    # print(str(T_value))
    # raise Exception()

    # training dataとverification dataの分離
    data_num = X.shape[0]
    data_num_t = data_num - cfg.verify_data_num
    data_locs = [i for i in range(data_num)]
    random.shuffle(data_locs)
    X_t = X[data_locs[: data_num_t]]
    X_v = X[data_locs[data_num_t:]]
    T_class_t = T_class[data_locs[: data_num_t]]
    T_class_v = T_class[data_locs[data_num_t:]]
    T_value_t = T_value[data_locs[: data_num_t]]
    T_value_v = T_value[data_locs[data_num_t:]]
    X_names = input_column_names()  # feature labels

    # K-Neareset Neighbors
    knn_handler = KNearestNeighborsHandler(cfg=cfg)
    accuracies_knn = knn_handler.training(
        X_train=X_t, T_train=T_value_t, X_verify=X_v, T_verify=T_value_v, X_column_names=X_names)
    knn_handler.show_log()
    knn_handler.export_log()

    # Logistic Regression
    logireg_handler = LogisticRegressionHandler(cfg=cfg)
    accuracies_logireg = logireg_handler.training(
        X_train=X_t, T_train=T_value_t, X_verify=X_v, T_verify=T_value_v, X_column_names=X_names)
    logireg_handler.show_log()
    logireg_handler.export_log()

    # SVC (Support Vector Classification)
    svc_handler = SVCHandler(cfg=cfg)
    accuracies_svc = svc_handler.training(
        X_train=X_t, T_train=T_value_t, X_verify=X_v, T_verify=T_value_v, X_column_names=X_names)
    svc_handler.show_log()
    svc_handler.export_log()

    # それぞれの性能をplot
    plot_performances(
        input_dict={
            "KNN": accuracies_knn,
            "LogiReg": accuracies_logireg,
            "SVC": accuracies_svc},
        path_dir=cfg.path_working_dir)

    # 最適化済みの各モデルをシリアライズして保存
    dill.dump(knn_handler, open(cfg.path_knn_handler, "wb"))
    dill.dump(logireg_handler, open(cfg.path_logireg_handler, "wb"))
    dill.dump(svc_handler, open(cfg.path_svc_handler, "wb"))



