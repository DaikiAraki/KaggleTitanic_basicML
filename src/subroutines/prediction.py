# Copyright 2022 Daiki Araki. All Rights Reserved.

import dill
from src.settings.Config import Config
from src.shared_functions.io_dataset import dataset_read_titanic_test
from src.models.KNearestNeighborsHandler import KNearestNeighborsHandler
from src.models.LogisticRegressionHandler import LogisticRegressionHandler
from src.models.SVCHandler import SVCHandler

"""
保存された最適化済みモデルを読みだしてKaggle提出用ファイルを作成する。
"""

def prediction(cfg):

    if cfg is None:
        cfg = Config()

    # titanic.zipからデータ読み出し
    X, I = dataset_read_titanic_test(path=cfg.path_titanic_zip, zip_mode=True, name="test.csv")

    # K-Nearest Neighbors
    with open(cfg.path_knn_handler, "rb") as f:
        knn_handler = dill.load(f)

    knn_handler.predict(X=X, I=I, make_submisson_csv=True)

    # Logistic Regression
    with open(cfg.path_logireg_handler, "rb") as f:
        logireg_handler = dill.load(f)

    logireg_handler.predict(X=X, I=I, make_submisson_csv=True)

    # SVC
    with open(cfg.path_svc_handler, "rb") as f:
        svc_handler = dill.load(f)

    svc_handler.predict(X=X, I=I, make_submisson_csv=True)



