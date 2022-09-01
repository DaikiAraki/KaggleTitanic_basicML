# Copyright 2022 Daiki Araki. All Rights Reserved.

from pathlib import Path
from datetime import datetime as dt
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.settings.Config import Config
from src.models.ModelHandlerBase import ModelHandlerBase

"""
Logistic Regression

事前処理はStandardizationを用いる。
正則化項の種類などのハイパーパラメータについてはGrid Searchで最適なものを探す。
"""

class LogisticRegressionHandler(ModelHandlerBase):

    def __init__(self, cfg, name="logireg"):
        """
        :param cfg: Config object
        :param name: str
        """
        assert isinstance(cfg, Config) and isinstance(name, str)
        super(LogisticRegressionHandler, self).__init__(cfg=cfg, name=name)
        self.preprocess = skl.preprocessing.StandardScaler()
        self.model = LogisticRegression(max_iter=10000)
        self.path_working_dir = cfg.path_logireg_dir
        self.path_working_dir.mkdir(parents=True, exist_ok=True)


    def training(self, X_train, T_train, X_verify, T_verify, X_column_names):
        """
        T is given by convert_T_value()
        :param X_train: training input data. [batch, width]
        :param T_train: training target data (Regression). [batch]
        :param X_verify: verification input data. [batch, width]
        :param T_verify: verification target data (Regression). [batch]
        :param X_column_names: column names of input data. list[width] of str
        :return losses: dict{"loss_train": scalar, "loss_verify": scalar}
        """
        print("=" * 10 + " LogisticRegressionHandler.training " + "=" * 10)
        time_start = dt.now()
        self.input_column_names = X_column_names

        # Grid Searchで最良のハイパーパラメータを選択
        params = [  # 探索対象のgridを作成
            {"penalty": ["none"],  # 正則化項
             "solver": ["lbfgs", "saga"]},  # 最適化手法
            {"penalty": ["l2"],
             "solver": ["lbfgs", "liblinear", "saga"],
             "C": [0.001, 0.01, 0.1, 1., 10.]},  # C: 正則化項の係数の逆数
            {"penalty": ["l1"],
             "solver": ["liblinear", "saga"],
             "C": [0.001, 0.01, 0.1, 1., 10.]}]

        clf = Pipeline(
            [("preprocess", self.preprocess),
             ("model",
              GridSearchCV(
                estimator=self.model,  # モデル
                param_grid=params,  # 探索範囲
                scoring="accuracy",  # 性能評価方法
                refit=True,  # 最後にestimatorをベストモデルに戻しておくかどうか
                cv=5,  # k-fold cross validation
                return_train_score=True))])

        clf.fit(X=X_train, y=T_train)
        self.model = clf.named_steps["model"].best_estimator_

        gs_cv = clf.named_steps["model"]
        self.log.update({
            "best_hyperparameters":
                {key: values[gs_cv.best_index_] for (key, values) in gs_cv.cv_results_.items() if key[:6] == "param_"},
            "best_parameters": self.get_parameters()})  # 最適パラメータを記録

        # 精度計算
        accuracy_train = accuracy_score(
            y_true=T_train, y_pred=self.model.predict(X=self.preprocess.transform(X=X_train)), normalize=True)
        accuracy_verify = accuracy_score(
            y_true=T_verify, y_pred=self.model.predict(X=self.preprocess.transform(X=X_verify)), normalize=True)

        # 総経過時間表示
        time_end = dt.now()
        print(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " " + "the training has finished. (elapsed time = " +
              f'{(time_end - time_start).seconds:,}' + "." +
              str((time_end - time_start).microseconds)[:3] + " [seconds])")

        accuracies = {"train": accuracy_train,
                      "verify": accuracy_verify}
        self.log.update(accuracies)  # 誤差値を記録

        print(dt.now().strftime("%Y/%m/%d %H:%M:%S") + " " + "all of the processes have done. <training>")

        return accuracies  # dict{"accuracy_train": scalar, "accuracy_verify": scalar}


    def predict(self, X, I, make_submisson_csv=True):
        """
        :param X: input data. [batch, width]
        :param I: index data. [batch]
        :param make_submisson_csv: bool. whether to make submisson csv file
        :return Y: output. [batch]
        """
        # 推定
        Y = self.model.predict(X=self.preprocess.transform(X=X))  # (0: 死亡, 1: 生存)の値として出てくる

        if make_submisson_csv:  # 推定値の保存
            self.export_submission(Y=Y, I=I)

        return Y  # [batch]


    def get_parameters(self):
        return {name: value for (name, value) in zip(self.input_column_names, self.model.coef_[0])}



