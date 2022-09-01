# Copyright 2022 Daiki Araki. All Rights Reserved.

from pathlib import Path
from datetime import datetime as dt
import numpy as np
from scipy import stats
import pandas as pd
import sklearn as skl
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from src.settings.Config import Config
from src.models.ModelHandlerBase import ModelHandlerBase

"""
SVC (Support Vector Classification)

事前処理はStandardizationを用いる。
ハイパーパラメータはRandomized Searchする。
収束が悪いケースがあり、その場合、sklearnのConvergenceWarningが出る。
非表示にしたければ、コメント化しているtraining()メソッドのデコレータをコメント化解除すればよい。
"""

class SVCHandler(ModelHandlerBase):

    def __init__(self, cfg, name="svc"):
        """
        :param cfg: Config object
        :param name: str
        """
        assert isinstance(cfg, Config) and isinstance(name, str)
        super(SVCHandler, self).__init__(cfg=cfg, name=name)
        self.preprocess = skl.preprocessing.StandardScaler()
        self.model = SVC(max_iter=10000000)
        self.path_working_dir = cfg.path_svc_dir
        self.path_working_dir.mkdir(parents=True, exist_ok=True)


    # @ignore_warnings(category=ConvergenceWarning)
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
        print("=" * 10 + " SVCHandler.training " + "=" * 10)
        time_start = dt.now()
        self.input_column_names = X_column_names

        # Randomized Searchで最良のハイパーパラメータを選択
        params = [  # 探索対象のgridを作成
            {"C": stats.loguniform(a=1.e-4, b=1.e2),  # 正則化項の係数の逆数
             "kernel": ["rbf", "sigmoid"],  # カーネル
             "gamma": stats.loguniform(a=1.e-3, b=1.e0)},  # カーネルの係数
            {"C": stats.loguniform(a=1.e-4, b=1.e2),
             "kernel": ["poly"],
             "degree": [3, 4],
             "gamma": stats.loguniform(a=1.e-3, b=1.e0)}]

        clf = Pipeline(
            [("preprocess", self.preprocess),
             ("model",
              RandomizedSearchCV(
                estimator=self.model,  # モデル
                param_distributions=params,  # 探索範囲
                n_iter=100,  # 試行回数
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
        return {}



