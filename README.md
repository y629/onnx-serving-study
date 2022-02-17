# onnx-serving-study
## 目的

ONNX Runtime Serving を用いた推論器の公開のお勉強．

参考: https://github.com/shibuiwilliam/ml-system-in-actions/tree/main/chapter4_serving_patterns

## 概要
- ImageNet で学習済みの InceptionV3 モデルを ONNX Runtime Serving で Serving する．
- クライアントは gRPC 通信で推論器サーバにリクエストを送る．
- ONNX Runtime Serving は推論しかやってくれないので，クライアント側で，画像データの前処理・後処理を行う．
- 推論器サーバから返ってきた推論結果はクライアント側で softmax で予測確率に変換して，予測ラベルを得る．

推論器サーバのためのコンテナ，クライアントを想定したコンテナをそれぞれ作り，Docker Compose で稼働する．


### ディレクトリ構成
```
.
├── README.md
├── Dockerfile.client (クライアント用のDockerfile)
├── Dockerfile.onnx_serving (推論器サーバ用のDockerfile)
├── docker-compose.yml
├── data (画像データとImageNetのラベルを保存)
│   ├──...
├── image_pred_request.py (クライアントの処理)
├── inception_v3_onnx_runtime (ONNX Runtime Servingの起動処理)
│   └──...
├── models (学習済みモデルファイル)
│   └──...
├── notebooks (お勉強に使ったnotebooks)
│   ├──...
├── proto (gRPC通信のために必要なメソッド群)
│   ├──...
├── requirements.txt
└── run.sh (コンテナの作成などのスクリプト)
```

## 使い方

0. カレントディレクトリ
```bash
$ pwd
~/onnx-serving-study
```

1. Docker イメージのビルドおよび Docker Compose を用いてサービスを稼働する．
```bash
$ bash run.sh
# 実行されるコマンド
# docker build -t onnx_serving_study_serving -f Dockerfile.onnx_serving .
# docker build -t onnx-serving-study_client -f Dockerfile.client .
# docker-compose up -d
```

2. クライアントを想定したコンテナに入り(抜ける時は`exit`)，推論器サーバへのリクエストを送り，推論結果を取得する．最終的に Top5 の予測確率と予測ラベルが表示される．
```bash
$ docker exec -it myclient /bin/bash
```

```bash
root@xxx:/onnx-serving-study＃ python image_pred_request.py
# 出力
# Loaded: ./data/dog.jpg
# request sending...(少し待ちます)
# response received...
# Top5 label prob
# Samoyed 0.8238476514816284
# Arctic fox 0.014191611669957638
# white wolf 0.012303538620471954
# Pomeranian 0.007874662056565285
# keeshond 0.006387784145772457
```

3. 画像ファイルのパス(`./data/cat.jpg` or `./data/siamese_cat.jpg` or `./data/dof.jpg`)を引数としてを与えることで好きな画像を推論できる．
```bash
root@xxx:/onnx-serving-study＃ python image_pred_request.py ./data/siamese_cat.jpg
# 出力
# Loaded: data/siamese_cat.jpg
# request sending...
# response received...
# Top5 label prob
# Siamese cat 0.8431887030601501
# lynx 0.020345453172922134
# Egyptian cat 0.020162323489785194
# tabby 0.0020235965494066477
# wallaby 0.0017176532419398427
```

4. Docker Compose を停止
```bash
$ docker-compose down
```