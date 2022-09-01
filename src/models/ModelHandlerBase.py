# Copyright 2022 Daiki Araki. All Rights Reserved.

import copy
import random
import zipfile
import inspect
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from datetime import datetime as dt
from src.settings.Config import Config
from src.shared_functions.io_dataset import dataset_read_titanic_train, dataset_read_titanic_test
from src.shared_functions.io_value import export_value

"""
【説明】
各モデルのHandlerクラスの共通部をまとめたもの。
"""

class ModelHandlerBase:

    def __init__(self, cfg, name):
        """
        :param cfg: Config object
        :param name: str
        """
        self.name = name
        self.verify_data_num = cfg.verify_data_num
        self.path_working_dir = cfg.path_working_dir
        self.model = None
        self.input_column_names = []
        self.log = {}


    def training(self, X_train, T_train, X_verify, T_verify, X_column_names):
        """
        :param X_train: training input data. [batch, width]
        :param T_train: training target data (Regression). [batch]
        :param X_verify: verification input data. [batch, width]
        :param T_verify: verification target data (Regression). [batch]
        :param X_column_names: column names of input data. list[width] of str
        :return losses: dict{"loss_train": scalar, "loss_verify": scalar}
        """
        raise Exception(
            "Implementation Error: method ModelHandlerBase.training() was called before implementation.")


    def predict(self, X, I, make_submisson_csv=True):
        """
        :param X: input data. [batch, width]
        :param I: index data. [batch]
        :param make_submisson_csv: bool. whether to make submisson csv file
        :return Y: output. [batch]
        """
        raise Exception(
            "Implementation Error: method ModelHandlerBase.inference() was called before implementation.")


    def get_parameters(self):
        raise Exception(
            "Implementation Error: method ModelHandlerBase.get_parameters() was called before implementation.")


    def export_submission(self, Y, I):
        """
        :param Y: output data. [batch]
        :param I: index data. [batch]
        """
        df = pd.DataFrame(
            data=np.concatenate([np.expand_dims(I, axis=1), np.expand_dims(Y, axis=1)], axis=1),
            dtype=int,
            index=None, columns=["PassengerId", "Survived"])
        df.to_csv(
            Path(str(self.path_working_dir) +
                 "\\submission_" + type(self.model).__name__ + "_" + self.name + ".csv"), index=False, header=True)
        self.export_log()


    def export_log(self):
        p = str(self.path_working_dir) + "\\log_" + type(self.model).__name__ + "_" + self.name + ".log"
        with open(p, "a") as f:
            f.write("=" * 40 + "\n")
            f.write("    class name: " + type(self.model).__name__ + "\n")
            f.write("    name attribute: " + self.name + "\n")
            f.write("    time: " + dt.now().strftime("%Y/%m/%d %H:%M:%S.%f") + "\n")
            f.write("-" * 40 + "\n")
            self.__export_log(f=f, d=self.log)
            f.write("\n")


    def __export_log(self, f, d, _current_indent_coef=0):
        """
        :param f: file object
        :param d: dict object
        :param _current_indent_coef: coef for "  ".
        """
        for (k, v) in d.items():
            f.write("  " * _current_indent_coef + str(k) + ": ")
            if isinstance(v, dict):
                f.write("\n")
                self.__export_log(f=f, d=v, _current_indent_coef=_current_indent_coef + 1)
            else:
                v_str = str(v) if not inspect.isclass(v) else type(v).__name__
                f.write(v_str + "\n")


    def show_log(self):
        print("=" * 40)
        print("    class: " + type(self.model).__name__)
        print("    name attribute: " + self.name)
        print("    time: " + dt.now().strftime("%Y/%m/%d %H:%M:%S.%f"))
        print("-" * 40)
        self.__show_log(d=self.log)
        print("")


    def __show_log(self, d, _current_indent_coef=0):
        """
        :param d: dict object
        :param _current_indent_coef: coef for "  ".
        """
        for (k, v) in d.items():
            if isinstance(v, dict):
                print(
                    "  " * _current_indent_coef + str(k) + ": ")
                self.__show_log(d=v, _current_indent_coef=_current_indent_coef + 1)
            else:
                print(
                    "  " * _current_indent_coef + str(k) + ": " +
                    (str(v) if not inspect.isclass(v) else type(v).__name__))



