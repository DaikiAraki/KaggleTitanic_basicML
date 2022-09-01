# Copyright 2022 Daiki Araki. All Rights Reserved.

from src.settings.Config import Config
from src.subroutines.training import training
from src.subroutines.prediction import prediction

"""
１：【実行方法】
　　(1) srcフォルダの存在する階層に（srcフォルダと並列に）titanic.zip（titanicで頒布されているデータセット）を配置する。
　　(2) このファイルを実行する。
２：srcフォルダの存在する階層に"data"というディレクトリが形成され、そこに結果等のデータが格納される。
"""


def main():

    cfg = Config()

    training(cfg=cfg)

    prediction(cfg=cfg)


if __name__ == "__main__":

    main()



