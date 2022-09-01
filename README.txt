author: 荒木大輝 (Araki Daiki)
e-mail: monochromaticstripes1838@gmail.com
date: 2022/09/01
========================

Kaggleの入門用課題であるtitanicについて、機械学習モデルで予測するプログラム。

課題がClassificationなので、手法はK-Nearest Neighbor, Logistic Regression, SVC (Support Vector Classification)を用いた。

以下の、Transformerユニットを用いたDeep Learningモデル：
https://github.com/DaikiAraki/KaggleTitanic_DeepLearning_Transformer
の時に比べて、良い結果が得られるかと思っていたが、３モデルともKaggle提出時の正解率は75%前後であり、ほぼ同じ性能であった。
あまりいい精度にならないのは、モデルの問題ではなく、ソースデータから入力データを抽出する部分の問題かもしれない。（データの作り方の問題で、元データに本来含まれている重要な情報が抽出されていない可能性。）

