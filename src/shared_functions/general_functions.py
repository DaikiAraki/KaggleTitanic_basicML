# Copyright 2022 Daiki Araki. All Rights Reserved.

import numpy as np
import tensorflow as tf

def argmax(a):

    # 引数の条件：１次元の list か tuple か np.ndarray である事
    assert isinstance(a, (list, tuple, np.ndarray)) and (not any(hasattr(a_i, "__iter__") for a_i in a))

    if isinstance(a, np.ndarray):
        return np.argmax(a=a)

    else:
        return a[a.index(max(a))]


def argmin(a):

    # 引数の条件：１次元の list か tuple か ndarray である事
    assert isinstance(a, (list, tuple, np.ndarray)) and (not any(hasattr(a_i, "__iter__") for a_i in a))

    if isinstance(a, np.ndarray):
        return np.argmin(a=a)

    else:
        return a[a.index(min(a))]


def sign(v):

    calc = lambda x: type(x)(1 if (x > 0) else (-1 if (x < 0) else 0))

    if type(v) in (list, tuple):
        obj_tp = type(v)
        return obj_tp((calc(x=x) for x in v))

    elif isinstance(v, np.ndarray):
        return np.sign(x=v)

    elif type(v) in (int, float):
        return calc(v)

    else:
        raise Exception(
            "ERROR: invalid argument type '" + str(type(v)) + "' in general_func.sign()\n"
            "type of argument 'v' must be in (int, float, list, tuple, np.ndarray)")


def minabs(a):

    # 引数の条件：１次元の list か tuple か ndarray
    assert isinstance(a, (list, tuple, np.ndarray)) and (not any(hasattr(a_i, "__iter__") for a_i in a))

    return min((abs(v) for v in a))


def flatten(data):
    # list か tuple について、flattenして返す
    return [
        b for a in data for b in (  # 二重forなので記述順に注意
            flatten(a) if (hasattr(a, "__iter__") and (not isinstance(a, str)))
            else (a,))]



