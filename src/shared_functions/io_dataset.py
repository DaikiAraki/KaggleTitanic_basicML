# Copyright 2022 Daiki Araki. All Rights Reserved.

import zipfile
import pandas as pd
from pathlib import Path
from src.shared_functions.titanic_data_conversion import convert_X, convert_T_class, convert_T_value, convert_I

def dataset_read_titanic_train(path, zip_mode=True, name="train.csv"):
    """
    :param path: pathlib.Path. path of train.csv or titanic.zip
    :param zip_mode: bool. whether the dataset file is in zip file.
    :param name: the file name to choose file if zip_mode is True
    :return X, T: [batch,width], [batch,2]. input-data, target-data
    """
    assert (isinstance(name, str) and (len(name) > 0)) if zip_mode else True

    if zip_mode:

        try:
            zf = zipfile.ZipFile(str(path), "r")
            f_train = zf.open(name, mode="r")
            pd_train = pd.read_csv(f_train, header=0, index_col=None)
            zf.close()

        except Exception as e:
            print(
                "ERROR: データが読み込めませんでした。\n" +
                "'" + str(path) + "' に Kaggle Titanic のzipデータ（titanic.zip）を配置してください。")
            raise e

    else:

        try:
            f_train = open(str(path), "r")
            pd_train = pd.read_csv(f_train, header=0, index_col=None)
            f_train.close()

        except Exception as e:
            print(
                "ERROR: データが読み込めませんでした。\n" +
                "ファイル '" + str(path) + "' が正しいtraining用データのcsvファイルであるかを確認してください。")
            raise e

    X = convert_X(data_rc=pd_train.values.tolist(), header=pd_train.columns.to_list())
    T_class = convert_T_class(data_rc=pd_train.values.tolist(), header=pd_train.columns.to_list())
    T_value = convert_T_value(data_rc=pd_train.values.tolist(), header=pd_train.columns.to_list())

    return X, T_class, T_value


def dataset_read_titanic_test(path, zip_mode=True, name="test.csv"):
    """
    :param path: pathlib.Path. path of test.csv or titanic.zip
    :param zip_mode: bool. whether the dataset file is in zip file.
    :param name: the file name to choose file if zip_mode is True
    :return X, I: [batch,width], [batch]. input-data, data-indices
    """
    assert (isinstance(name, str) and (len(name) > 0)) if zip_mode else True

    if zip_mode:

        try:
            zf = zipfile.ZipFile(str(path), "r")
            f_test = zf.open("test.csv", mode="r")
            pd_test = pd.read_csv(f_test, header=0, index_col=None)
            zf.close()

        except Exception as e:
            print(
                "ERROR: データが読み込めませんでした。\n" +
                "'" + str(path) + "' に Kaggle Titanic のzipデータ（titanic.zip）を配置してください。")
            raise e

    else:

        try:
            f_test = open(str(path), "r")
            pd_test = pd.read_csv(f_test, header=0, index_col=None)
            f_test.close()

        except Exception as e:
            print(
                "ERROR: データが読み込めませんでした。\n" +
                "ファイル '" + str(path) + "' が正しいtest用データのcsvファイルであるかを確認してください。")
            raise e

    X = convert_X(data_rc=pd_test.values.tolist(), header=pd_test.columns.to_list())
    I = convert_I(data_rc=pd_test.values.tolist(), header=pd_test.columns.to_list())

    return X, I



