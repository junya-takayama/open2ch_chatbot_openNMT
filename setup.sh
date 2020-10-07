# 対話コーパスのダウンロード
git clone https://github.com/1never/open2ch-dialogue-corpus
cd open2ch-dialogue-corpus/data
wget http://keldic.net/data/open2ch_dialogue_corpus.zip
unzip open2ch_dialogue_corpus.zip
cd ../
python ./cleaning.py --input_file ./data/livejupiter.tsv --output_file ./data/livejupiter_cleaned.tsv
cd ../

# OpenNMT-py のインストール
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
python setup.py install
cd ../

# sentencepiece のインストール
pip install sentencepiece