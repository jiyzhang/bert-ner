
from pathlib import Path
import pickle
import numpy as np
from collections import Counter

DATADIR = Path("./data/wind")
def words(name):
    return '{}.words.txt'.format(name)

def tags(name):
    return '{}.tags.txt'.format(name)

def fwords(name):
    return str(Path(DATADIR, '{}.words.txt'.format(name)))

def ftags(name):
    return str(Path(DATADIR, '{}.tags.txt'.format(name)))

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-ORG": 3, "I-ORG": 4
             }



def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data

def vocab_build(vocab_path, corpus_path, min_count):
    """
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with Path(vocab_path).open('wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """
    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
    :param vocab_path:
    :return:
    """
    #vocab_path = os.path.join(vocab_path)
    with Path(vocab_path).open("rb") as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab_size, embedding_dim):
    """
    :param vocab_size:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def build_hanzi_vocab():
    # 1. 汉字
    # Get Counter of 汉字 on all the data, filter by min count, save
    MINCOUNT = 1

    print('Building vocab for Hanzi ')
    counter_words = Counter()
    for n in ['train', 'valid', 'test']:
        with Path(fwords(n)).open() as f:
            for line in f:
                counter_words.update(line.strip().split('|'))

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path(DATADIR / 'vocab.words.txt').open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(len(vocab_words), len(counter_words)))

def build_tag_vocab():
    # 2. Tags
    # Get all tags from the training set

    print('Build vocab for tags')
    vocab_tags = set()
    with Path(ftags('train')).open() as f:
        for line in f:
            vocab_tags.update(line.strip().split('|'))

    with Path(DATADIR / 'vocab.tags.txt').open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))

def build_hanzi_embedding():
    # Load vocab
    with Path(DATADIR / 'vocab.words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant char Chinese vectors
    found = 0
    print('Reading Chinese Char Vectors (may take a while)')
    with Path(DATADIR / 'sgns.context.word-character.char1-1.dynwin5.thr10.neg5.dim300.iter5').open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split('|')
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed(DATADIR / 'sgns.npz', embeddings=embeddings)