# chemdner_pytorch

## ファイルの説明
* chemdner_data_converter.py
    * chendnerのデータをTagging用のデータに変換するためのModule
    * train.csv, valid.csv, test.csvを作成。
    * それぞれのcsvのheaderは[file_ix, token, start_ix, end_ix, label]
* dataset.py
    * token2ix, label2ix用
* train.py
    * modelを学習しsave
* model.py
    * modelの計算方法を書く。
* predict.py
    * test.csvから予測し、test.csvにpred_label列を付け合わせたpred.csvを作成
* evaluate.py
    * pred_labelからspantokenを復元し、精度を計算する。

* chemdner_datas
    * chemdnerdataからconvertしたcsvやtoken2ix, label2ixの入れ場
* outsource
    * 学習したモデルの入れ場


# 実験方法
```
# これは初めだけ行えば良い。
>>> python chemdner_data_converter.py
# modelを学習し、予測し、評価
>>> python train.py
>>> python predict.py
>>> python evaluate.py
```
