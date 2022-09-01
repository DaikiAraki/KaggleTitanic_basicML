# Copyright 2022 Daiki Araki. All Rights Reserved.

import csv
import traceback
import copy
import time
import numpy as np
from datetime import datetime as dt

def export_value(path, data_list, t=None, header=None):

    """
    値の出力関数
    pathで指定したファイルに、csvの追記モードで書き込みをする
    :param path: pathlib.Path object
    :param data_list: flat(1d) list or tuple or int, float, datetime.datetime, str
    :param t: time as additional data
    :param header: header row, list of str
    """

    values = copy.deepcopy(data_list)
    
    try:
        if isinstance(values, list):
            pass
        elif isinstance(values, (np.ndarray, np.float32, np.float64)):
            values = values.tolist()
        elif isinstance(values, (float, int, dt, str)):
            values = [values]
        elif isinstance(values, tuple):
            values = list(values)
    except:
        print(traceback.format_exc())
    
    flag = False
    
    if not path.exists():
        flag = True

    count = 0
    while count < 1200:  # wait up to 2 minutes
        try:
            with open(str(path), "a") as f:
                writer = csv.writer(f, lineterminator="\n")

                if flag and (header is not None):
                    writer.writerow((header + ["time"]) if (t is not None) else header)

                writer.writerow((values + [t]) if (t is not None) else values)
                break

        except:
            print(traceback.format_exc())

        count += 1
        time.sleep(0.500)

    if count >= 1200:
        print("ERROR: gave up to retring export_value()")

    del values
    


