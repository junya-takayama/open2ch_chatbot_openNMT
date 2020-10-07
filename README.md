# OpenNMT-py を用いた雑談対話システム
このリポジトリはコトバデザイン様主催の[対話システム勉強会（第一回）](https://cotobaagent-developers-community.connpass.com/event/188047/) 3コマ目 "OpenNMTによる雑談対話機能の実装" で使用する教材の一部です．  
[おーぷん2ちゃんねる対話コーパス](https://github.com/1never/open2ch-dialogue-corpus) を学習データとして，OpenNMT-py を用いて Seq2Seq ベースの対話システムを構築します．

[講義スライド](https://github.com/junya-takayama/open2ch_chatbot_openNMT/blob/main/slides.pdf)  
[Colab notebook](https://colab.research.google.com/drive/1Fs-wklGpXaew2KBAowBXbeYEDycDMsly?usp=sharing)

## 使いかた
* Rest API サーバの構築以外は [Colab](https://colab.research.google.com/drive/1Fs-wklGpXaew2KBAowBXbeYEDycDMsly?usp=sharing) でも動かせるようにしています．説明も Colab の方が詳しいです．  
* 手っ取り早く Rest API サーバを立ち上げたい方は[学習済みモデル一式](https://drive.google.com/file/d/1nVoH6GJx4f7D2UcQUSv_uRiAVq4UBW4v/view?usp=sharing)を trained_model 以下に配置し，手順4に飛んでください．  
* config ファイルをいくつか用意したので，余力があれば試してみてください（config_chatbot_livejupiter_transformer.yaml はうまくチューニングしないとまともに学習しないかもしれません）

### 手順1: 事前準備
おーぷん2ちゃんねるコーパスのダウンロードと OpenNMT-py のインストールを行います  
```sh
bash setup.sh
```

### 手順2: おーぷん2ちゃんねるコーパスの前処理
sentencepiece を用いてコーパス全体をトークナイズし，OpenNMT-py 準拠の形式で保存します．  
```sh
python build_corpus_and_tokenizer.py
```

### 手順3: 対話システムの構築
#### 語彙辞書の構築
```sh
onmt_build_vocab --config config_chatbot_livejupiter.yaml -n_sample 100000
```

#### 訓練
必ず GPU 環境で（なければ Colab で）  
```sh
onmt_train --config config_chatbot_livejupiter.yaml
```

#### テストデータでの推論
Beam Search (k=5)  
`<任意のステップ数>` とある部分については，基本的には valid での ppl が一番低かったステップ数を選ぶと良いです
```sh
onmt_translate --model ./trained_model/model_chatbot_livejupiter_step_<任意のステップ数>.pt --src ./data/test.src --output data/pred_beam.txt --gpu 0 --replace_unk
```  

Topk-Sampling (k=5)
```sh
onmt_translate --model ./trained_model/model_chatbot_livejupiter_step_<任意のステップ数>.pt --src ./data/test.src --output data/pred_sampling.txt --gpu 0 --replace_unk --random_sampling_topk 5
```

### 手順4: Rest API サーバの構築
rest_config.json の `"model": "model_chatbot_livejupiter_step_103500.pt",` は適宜書き換えてください．  
以下のコマンドを実行すると localhost:5000 で Rest API サーバが立ち上がります
```sh
python server.py --config rest_config.json
```
`/translator/translate` に POST で以下のようにテキストを送ると，
```python
import requests
import json

headers = {"Content-Type" : "application/json"}
contents = json.dumps([{"src": "野球したい", "id": 0}])
print(requests.post("http://localhost:5000/translator/translate, contents, headers=headers).json())
```
以下のような response が帰ってきます．`"tgt"` がシステムからの応答になります．
```python
[[{'n_best': 1, 'pred_score': -0.5358448028564453, 'src': '野球したい', 'tgt': 'やきう興味ないわ'}]]
```
