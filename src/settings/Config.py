# Copyright 2022 Daiki Araki. All Rights Reserved.

from pathlib import Path

class Config:

    def __init__(self):
        self.verify_data_num = 100
        self.path_working_dir = None
        self.path_titanic_zip = None
        self.path_knn_dir = None
        self.path_logireg_dir = None
        self.path_svc_dir = None
        self.path_knn_handler = None
        self.path_logireg_handler = None
        self.path_svc_handler = None
        self.set_path_working_dir(path_working_dir=Path(str(Path.cwd()) + "\\data"))

    def set_path_working_dir(self, path_working_dir):
        self.path_working_dir = Path(str(path_working_dir))
        self.path_working_dir.mkdir(parents=True, exist_ok=True)
        self.path_titanic_zip = Path(str(Path.cwd()) + "\\titanic.zip")
        self.path_knn_dir = Path(str(path_working_dir) + "\\knn")  # for K-Nearest Neighbors
        self.path_logireg_dir = Path(str(path_working_dir) + "\\logireg")  # for Logistic Regression
        self.path_svc_dir = Path(str(path_working_dir) + "\\svc")  # for SVC
        self.path_knn_handler = Path(str(self.path_knn_dir) + "\\knn_handler.dill")
        self.path_logireg_handler = Path(str(self.path_logireg_dir) + "\\logireg_handler.dill")
        self.path_svc_handler = Path(str(self.path_svc_dir) + "\\svc_handler.dill")

    def set_verify(self, num):
        self.verify_data_num = num



