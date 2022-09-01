# Copyright 2022 Daiki Araki. All Rights Reserved.

import numpy as np

def input_column_names():
    names = [
        "Pclass",
        "Name_normal", "Name_high",
        "Sex_male", "Sex_female",
        "Age",
        "SibSp",
        "Fare",
        "Cabin_A", "Cabin_B", "Cabin_C", "Cabin_D", "Cabin_E", "Cabin_F", "Cabin_G", "Cabin_other",
        "Embarked_C", "Embarked_Q", "Embarked_S"
    ]
    return names

def output_class_names():
    names = [
        "Dead", "Alive"
    ]
    return names

def convert_X(data_rc, header):
    """
    モデルに入力するデータを作成
    :param data_rc: 2d list, [row, column] = [batch, width]
    :param header: 1d list, corresponds to data_rc's column
    :return: 2d np.ndarray, [batch, width]
    """
    # data is a transposed array of data_rc
    data = [[] for i in range(len(header))]
    for i in range(len(data_rc)):

        if len(data_rc[i]) == len(header) - 1:  # 抜け補完
            data_rc[i].append("")

        for j in range(len(header)):
            data[j].append(data_rc[i][j])

    # make a list of 2d np.ndarray
    dataList = [
        convert_Pclass(data=data[header.index("Pclass")]),
        convert_Name(data=data[header.index("Name")]),
        convert_Sex(data=data[header.index("Sex")]),
        convert_Age(data=data[header.index("Age")]),
        convert_SibSp(data=data[header.index("SibSp")]),
        convert_Parch(data=data[header.index("Parch")]),
        convert_Fare(data=data[header.index("Fare")]),
        convert_Cabin(data=data[header.index("Cabin")]),
        convert_Embarked(data=data[header.index("Embarked")])]

    X = np.concatenate(dataList, axis=0).T
    return X  # np.ndarray, [batch, width]

def convert_T_class(data_rc, header):
    """
    ターゲット値をclassification出力の形式で得る
    :param data_rc: 2d list, [row, column]
    :param header: 1d list, corresponds to data_rc's column
    :return: 2d np.ndarray, [batch, width]
    """
    # data is a transposed array of data_rc
    data = [[] for i in range(len(header))]
    for i in range(len(data_rc)):

        if len(data_rc[i]) == len(header) - 1:  # 抜け補完
            data_rc[i].append("")

        for j in range(len(header)):
            data[j].append(data_rc[i][j])

    T = convert_Survived_class(data=data[header.index("Survived")]).T
    return T  # np.ndarray, [batch, width]

def convert_T_value(data_rc, header):
    """
    ターゲット値をregression出力の形式で得る
    :param data_rc: 2d list, [row, column]
    :param header: 1d list, corresponds to data_rc's column
    :return: np.ndarray, [batch]
    """
    # data is a transposed array of data_rc
    data = [[] for i in range(len(header))]
    for i in range(len(data_rc)):

        if len(data_rc[i]) == len(header) - 1:  # 抜け補完
            data_rc[i].append("")

        for j in range(len(header)):
            data[j].append(data_rc[i][j])

    T = convert_Survived_value(data=data[header.index("Survived")]).T
    return T  # np.ndarray, [batch]

def convert_I(data_rc, header):
    """
    データIDを抽出
    :param data_rc: 2d list, [row, column]
    :param header: 1d list, corresponds to data_rc's column
    :return: 1d np.ndarray, [batch]
    """
    # data is a transposed array of data_rc
    data = [[] for i in range(len(header))]
    for i in range(len(data_rc)):

        if len(data_rc[i]) == len(header) - 1:  # 抜け補完
            data_rc[i].append("")

        for j in range(len(header)):
            data[j].append(data_rc[i][j])

    I = np.array(data[header.index("PassengerId")], dtype=np.int32)
    return I  # np.ndarray, [batch]

def convert_Survived_class(data):  # channel=2
    """
    生存したかどうか（教師値）をclassification出力の形式で得る
    （0: 死，1: 生）を [死亡, 生存] の形式で、当てはまる方で 1、それ以外は 0 を取るnp.ndarrayにする
    :param data: Survived, 1d list
    :return: [batch, 2 (=dead,alive)]
    """
    res = []
    for i in range(len(data)):
        if data[i] == 0:
            res.append([1, 0])
        elif data[i] == 1:
            res.append([0, 1])
        else:
            raise Exception(
                "ERROR: detect missing data in [model.modelFunction.dataConversion.convert_Survived_class]\n" +
                "argument 'data'[" + str(i) + "] = " + str(data[i]))

    return np.array(res, dtype=np.float32).T  # [batch, 2]

def convert_Survived_value(data):  # channel=1
    """
    生存したかどうか（教師値）をregression出力の形式で得る
    :param data: Survived, 1d list
    :return: [batch, survived]
    """
    return np.array(data, dtype=np.float32).T  # [batch]

def convert_Pclass(data):  # channel=1
    """
    乗客のチケットクラスを数値化
    「良い」方が値が大きくなるように
    :param data: Pclass, 1d list
    :return: [batch, pclass]
    """
    res = []
    for i in range(len(data)):
        if data[i] == 1:
            res.append([3])
        elif data[i] == 2:
            res.append([2])
        elif data[i] == 3:
            res.append([1])
        else:
            raise Exception(
                "ERROR: encountered invalid value in [model.modelFunction.dataConversion.convert_Pclass()]\n" +
                "argument 'data' = " + str(data))

    return  np.array(res, dtype=np.float32).T  # [batch, 1]

def convert_Name(data):  # channel=2
    """
    記載氏名から敬称が有るか無いかをクラス化
    :param data: Name
    :return: [batch, 2(=normal,high)]
    """
    res = []
    for i in range(len(data)):
        if data[i] in [np.nan]:
            res.append([0, 1])
        elif (("Mr." in data[i]) or ("Miss." in data[i]) or ("Ms." in data[i]) or
                ("Mrs." in data[i]) or ("Mme." in data[i]) or ("Mlle." in data[i])):
            res.append([1, 0])
        else:
            res.append([0, 1])
    return np.array(res, dtype=np.float32).T  # [batch, 2]

def convert_Sex(data):  # channel=2
    """
    性別をクラス化
    :param data: Sex
    :return: [batch, 2(=male,female)]
    """
    res = []
    for i in range(len(data)):
        if data[i] == "male":
            res.append([1, 0])
        elif data[i] == "female":
            res.append([0, 1])
        else:
            res.append([0, 0])
    return np.array(res, dtype=np.float32).T  # [batch, 2]

def convert_Age(data):  # channel=1
    """
    年齢
    :param data: Age
    :return: [batch, age]
    """
    res = np.array([data], dtype=np.float32)
    res[np.isnan(res)] = 0.
    return res  # [batch, 1]

def convert_SibSp(data):  # channel=1
    """
    一緒に乗船している siblings or spouses の数
    :param data: SibSp
    :return: [batch, sibsp]
    """
    res = np.array([data], dtype=np.float32)
    res[np.isnan(res)] = 0.
    return res  # [batch, 1]

def convert_Parch(data):  # channel=1
    """
    一緒に乗船している parents or children の数
    :param data: Parch
    :return: [batch, parch]
    """
    res = np.array([data], dtype=np.float32)
    res[np.isnan(res)] = 0.
    return res  # [batch, 1]

def convert_Fare(data):  # channel=1
    """
    運賃
    :param data: Fare
    :return: [batch, fare]
    """
    res = np.array([data], dtype=np.float32)
    res[np.isnan(res)] = 0.
    return res  # [batch, 1]

def convert_Cabin(data):  # channel=8
    """
    客室番号の英数字をクラス化
    :param data: Cabin
    :return: [batch, 8(=A,B,C,D,E,F,G,ELSE)]
    """
    res = []
    for i in range(len(data)):
        if data[i] in ["", np.nan]:
            res.append([0, 0, 0, 0, 0, 0, 0, 0])
        else:
            r = [0, 0, 0, 0, 0, 0, 0, 0]
            if "A" in data[i]:
                r[0] = 1
            if "B" in data[i]:
                r[1] = 1
            if "C" in data[i]:
                r[2] = 1
            if "D" in data[i]:
                r[3] = 1
            if "E" in data[i]:
                r[4] = 1
            if "F" in data[i]:
                r[5] = 1
            if "G" in data[i]:
                r[6] = 1
            if sum(r) == 0:
                r[7] = 1
            res.append(r)
    return np.array(res, dtype=np.float32).T  # [batch, 8]

def convert_Embarked(data):  # channel=3
    """
    乗船港をクラス化
    :param data: Embarked
    :return: [batch, 3(=C,Q,S)]
    """
    res = []
    for i in range(len(data)):
        r = [0, 0, 0]
        if data[i] in [np.nan]:
            pass
        elif "C" in data[i]:
            r[0] = 1
        elif "Q" in data[i]:
            r[1] = 1
        elif "S" in data[i]:
            r[2] = 1
        res.append(r)
    return np.array(res, dtype=np.float32).T  # [batch, 3]



