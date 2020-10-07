# OpenNMT-py を用いた雑談対話システム
このリポジトリはコトバデザイン様主催の[対話システム勉強会（第一回）](https://cotobaagent-developers-community.connpass.com/event/188047/) 3コマ目 "OpenNMTによる雑談対話機能の実装" で使用する教材の一部です．  
[おーぷん2ちゃんねる対話コーパス](https://github.com/1never/open2ch-dialogue-corpus) を学習データとして，OpenNMT-py を用いて Seq2Seq ベースの対話システムを構築します．

## 使いかた
Rest API サーバの構築以外は [Colab]() でも動かせるようにしています．説明も Colab の方が詳しいです．  
手っ取り早く Rest API サーバを立ち上げたい方は[学習済みモデル一式]()を trained_model 以下に配置し，手順4に飛んでください．  
config ファイルをいくつか用意したので，余力があれば試してみてください（config_chatbot_livejupiter_transformer.yaml はうまくチューニングしないとまともに学習しないかもしれません）

### 手順1: 事前準備
おーぷん2ちゃんねるコーパスのダウンロードと OpenNMT-py のインストールを行います  
```
bash setup.sh
```

### 手順2: おーぷん2ちゃんねるコーパスの前処理
sentencepiece を用いてコーパス全体をトークナイズし，OpenNMT-py 準拠の形式で保存します．  
```
python build_corpus_and_tokenizer.py
```

### 手順3: 対話システムの構築
#### 語彙辞書の構築
```
onmt_build_vocab --config config_chatbot_livejupiter.yaml -n_sample 100000
```

#### 訓練
必ず GPU 環境で（なければ Colab で）  
```
onmt_train --config config_chatbot_livejupiter.yaml
```

#### テストデータでの推論
Beam Search (k=5)  
`<任意のステップ数>` とある部分については，基本的には valid での ppl が一番低かったステップ数を選ぶと良いです
```
onmt_translate --model ./trained_model/model_chatbot_livejupiter_step_<任意のステップ数>.pt --src ./data/test.src --output data/pred_beam.txt --gpu 0 --replace_unk
```  

Topk-Sampling (k=5)
```
!onmt_translate --model ./trained_model/model_chatbot_livejupiter_step_39500.pt --src ./data/test.src --output data/pred_sampling.txt --gpu 0 --replace_unk --random_sampling_topk 5
```

### 手順4: Rest API サーバの構築
rest_config.json の `"model": "model_chatbot_livejupiter_step_103500.pt",` は適宜書き換えてください．  
```
python server.py --config rest_config.json
```
