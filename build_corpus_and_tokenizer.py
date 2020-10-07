import sentencepiece as spm
import random

num_data = 1000000
n_valid = 10000
n_test = 10000


def read_corpus(data_path, num_data=-1):
    dialogue_corpus = []
    for i, line in enumerate(open(data_path)):
        dialogue = line.strip().split("\t") # [<発話1>, <発話2>, <発話3>, ...] 形式の可変長リスト 
        dialogue_corpus.append(dialogue) 
        if i == num_data: break
    return dialogue_corpus


def train_valid_test_divide(dialogue_corpus, n_valid=10000, n_test=10000):
    train_dialogue_corpus = dialogue_corpus[:- n_valid - n_test]
    valid_dialogue_corpus = dialogue_corpus[- n_valid - n_test: - n_test]
    test_dialogue_corpus = dialogue_corpus[- n_test: ]
    return {
        "train": train_dialogue_corpus, 
        "valid": valid_dialogue_corpus,
        "test": test_dialogue_corpus,
    }


def create_tokenized_parallelcorpus(dialogue_corpus, sp):
    utterances = []
    replies = []
    for dialogue in dialogue_corpus:
        tokenized_dialogue = [tokenize(text, sp) for text in dialogue]
        for i in range(len(dialogue) - 1):
            utterance = tokenized_dialogue[i]
            reply = tokenized_dialogue[i+1]
            utterances.append(utterance)
            replies.append(reply)
    return {"src": utterances, "tgt": replies}


def tokenize(text, sp):
    return " ".join(sp.EncodeAsPieces(text))

print("loading raw data...")
corpus_divided = train_valid_test_divide(
    read_corpus("./open2ch-dialogue-corpus/data/livejupiter_cleaned.tsv", num_data=num_data)
)

print("building training data for sentencepiece")
# sentencepiece 学習用データの作成 (学習にかなり時間がかかるので10万対話に限定)
data_for_spm = ""
for dialogue in random.sample(corpus_divided["train"],100000):
    for text in dialogue:
        data_for_spm += text + "\n"
open("./data/data_for_spm.txt", "w").write(data_for_spm)


# sentencepiece 学習
print("training sentencepiece model...")
sp = spm.SentencePieceProcessor()
spm.SentencePieceTrainer.Train(
    "--input=./data/data_for_spm.txt --model_prefix=./data/spm_trained_model --vocab_size=16000 --normalization_rule_name=identity --model_type=unigram  --max_sentence_length=2048"
)

# sentencepiece 学習済みモデルの読み込み
print("loading the trained sentencepiece model...")
sp = spm.SentencePieceProcessor()
sp.load("./data/spm_trained_model.model")

# パラレルコーパスの作成
print("building corpus for onmt...")
for corpus_type, corpus in corpus_divided.items():
    parallel_corpus = create_tokenized_parallelcorpus(corpus, sp)
    src = "\n".join(parallel_corpus["src"])
    tgt = "\n".join(parallel_corpus["tgt"])
    open("./data/"+corpus_type+".src", "w").write(src)
    open("./data/"+corpus_type+".tgt", "w").write(tgt)
    
print("done!")