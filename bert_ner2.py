#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.

0508, cp bert_bilstm_crf_ner2.py ../bert-ner/bert_ner2.py
"""

import collections
import os

import pickle
import codecs
import logging

import tensorflow as tf
import tf_metrics
from bert import modeling
from bert import optimization
from bert import tokenization
from conlleval_tpu import return_report

# from lstm_crf_layer import BiLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers


flags = tf.flags

FLAGS = flags.FLAGS
## Requried parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 256, #128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

### for LSTM_CRF
flags.DEFINE_float('keep_prob', 0.9,
                    "LSTMP keep_prob")
flags.DEFINE_integer('lstm_size', 128,
                    "size of lstm hidden units")
flags.DEFINE_integer('num_layers', 1,
                    'number of rnn layers, default is 1.')
tf.flags.DEFINE_string('cell', 'lstm',
                    'Cell Type (LSTM or GRU) used.')
tf.flags.DEFINE_bool('crf_only', True, 'whether just only CRF layer')

### for dataset format difference
tf.flags.DEFINE_string('datasetformat', 'wind', "dataset format, conll or wind")

tf.logging.set_verbosity(logging.DEBUG)

input_file_col_sep = ' '
example_col_sep = '|'

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask
        self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        # tf.logging.info("datasetformat: %s" % (FLAGS.datasetformat))
        # tf.logging.info("dataset format: conll")
        """Reads a BIO data."""
        #with codecs.open(input_file, 'r', encoding='utf-8') as f:
        with tf.gfile.Open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                if len(contends) == 0:
                    # l = ' '.join([label for label in labels if len(label) > 0])
                    # w = ' '.join([word for word in words if len(word) > 0])
                    l = example_col_sep.join([label for label in labels if len(label) > 0])
                    w = example_col_sep.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                elif contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                else:
                    # tokens = contends.split(' ')
                    tokens = contends.split(input_file_col_sep)
                    if len(tokens) == 2:
                        words.append(tokens[0])
                        labels.append(tokens[1])
                    elif len(tokens) == 1:    #<空格> <label>
                        words.append(' ')
                        labels.append(tokens[0])

            return lines

class NerProcessor(DataProcessor):
    # def __init__(self, output_dir):
    #     self.labels = set()
    #     self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        if FLAGS.datasetformat == "conll":
            return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        else:
            return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

# def write_tokens(tokens, output_dir, mode):
#     """
#     将序列解析结果写入到文件中
#     只在mode=test的时候启用
#     :param tokens:
#     :param mode:
#     :return:
#     """
#     if mode == "test":
#         path = os.path.join(output_dir, "token_" + mode + ".txt")
#         wf = codecs.open(path, 'a', encoding='utf-8')
#         for token in tokens:
#             if token != "**NULL**":
#                 wf.write(token + '\n')
#         wf.close()

def write_tokens(tokens,mode):
    if mode=="test":
        path = os.path.join(FLAGS.output_dir, "token_"+mode+".txt")
        openmode = ""
        if tf.gfile.Exists(path):
            openmode = 'a'
        else:
            openmode = 'w'
        wf = tf.gfile.Open(path, openmode)
        for token in tokens:
            if token!="**NULL**":
                wf.write(token+'\n')
        wf.close()

## for CoNLL data, the seperator is space
## for Wind Data, the seperator is `|`

def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_map: 标签列表 (label -> index)
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * max_seq_length, #label_ids=0,
            is_real_example=False)

    # if FLAGS.datasetformat == "wind":
    #     seperator = '|'
    # else:
    #     seperator = ' '

    textlist = example.text.split(example_col_sep)
    labellist = example.label.split(example_col_sep)
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），
        # 可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]

    ntokens     = []
    segment_ids = []
    label_ids   = []

    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病

    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])

    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # Zero-Pad up to the sentence length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True
        # label_mask = label_mask
    )
    # mode='test'的时候才有效
    # write_tokens(ntokens, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """

    label_map = {}
    # {'O': 1, 'B-PER': 2, 'I-PER': 3, 'B-ORG': 4, 'I-ORG': 5, 'X': 6, '[CLS]': 7, '[SEP]': 8}
    # starting from 1
    for (i, label) in enumerate(label_list,1):
        label_map[label] = i
    with tf.gfile.Open(os.path.join(FLAGS.output_dir, 'label2id.pkl'),'wb') as w:
        pickle.dump(label_map,w)

    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 对于每一个训练样本,
        # convert_single_example自动对长句子进行裁剪，到[CLS]max_seq_len - 2[SEP]
        feature = convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer,  mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, is_eval, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),  # seq_length
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)

        # if is_eval:
        #     # to avlid "end of sequence" error @eval stage
        #     d = d.repeat()

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                #num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=8)   ##//?
        return d

    return input_fn



def create_model_old(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            # Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
            output_layer = tf.nn.dropout(output_layer, rate = 0.1) #keep_prob=0.9)

        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        # probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        predict_ids = tf.argmax(logits, axis=-1)

        return (loss, per_example_loss, logits, predict_ids)


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings,
                 crf_only=True, keep_prob=0.9, lstm_size=1, cell='lstm', num_layers=1):
    """
    创建X模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :param crf_only: 是否只使用CRF
    :param keep_prob:
    :param lstm_size: hidden size of LSTM
    :param cell LSTM| GRU
    :param num_layers BiLSTM layes
    :return:
    """

    return create_model_old(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings)

    # # 使用数据加载BertModel,获取对应的字embedding
    # model = modeling.BertModel(
    #     config=bert_config,
    #     is_training=is_training,
    #     input_ids=input_ids,
    #     input_mask=input_mask,
    #     token_type_ids=segment_ids,
    #     use_one_hot_embeddings=use_one_hot_embeddings
    # )
    # # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    # embedding = model.get_sequence_output()
    # max_seq_length = embedding.shape[1].value
    # # 算序列真实长度
    # used = tf.sign(tf.abs(input_ids))
    # lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    # # 添加CRF output layer
    # blstm_crf = BiLSTM_CRF(embedded_chars=embedding, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
    #                        keep_prob=keep_prob, initializers=initializers, num_labels=num_labels,
    #                        seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    # rst = blstm_crf.add_bilstm_crf_layer(crf_only)
    # return rst

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):

    """
    Returns `model_fn` closure for TPUEstimator.

    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param use_one_hot_embeddings:
    :param args: ?
    :return:
    """

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_real_example = None  #$$$$ need to confirm
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        print('shape of input_ids', input_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        # total_loss, logits, trans, pred_ids = create_model(
        #     bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        #     num_labels, use_one_hot_embeddings,
        #     FLAGS.crf_only, FLAGS.keep_prob, FLAGS.lstm_size, FLAGS.cell, FLAGS.num_layers)

        (total_loss,  per_example_loss,logits,predit_ids) = create_model(
            bert_config, is_training, input_ids, input_mask,segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)


        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None

        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            #train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)
            train_op = optimization.create_optimizer(
                 total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            # hook_dict = {}
            # hook_dict['loss'] = total_loss
            # hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            # logging_hook = tf.train.LoggingTensorHook(
            #     hook_dict, every_n_iter=args.save_summary_steps)
            #
            # output_spec = tf.estimator.EstimatorSpec(
            #     mode=mode,
            #     loss=total_loss,
            #     train_op=train_op,
            #     training_hooks=[logging_hook])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            # def metric_fn(label_ids, pred_ids):
            #     return {
            #         "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
            #     }
            #
            # eval_metrics = metric_fn(label_ids, pred_ids)
            # output_spec = tf.estimator.EstimatorSpec(
            #     mode=mode,
            #     loss=total_loss,
            #     eval_metric_ops=eval_metrics
            # )

            def metric_fn(label_ids, predict_ids):
            # def metric_fn(label_ids, logits):
                # CRF已经解码过了
                #predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                predictions = predict_ids
                # print("predictions shape: " + str(predictions.get_shape().as_list()))
                # print("label_ids shape: " + str(label_ids.get_shape().as_list()))
                # print("is_real_example shape: " + str(is_real_example.get_shape().as_list()))
                precision = tf_metrics.precision(label_ids,predictions,num_labels,[2,3,4,5], average="macro")
                recall = tf_metrics.recall(label_ids,predictions,num_labels,[2,3,4,5], average="macro")
                f = tf_metrics.f1(label_ids,predictions,num_labels,[2,3,4,5], average="macro")

                # precision = tf_metrics.precision(label_ids, predictions, 11, [2, 3, 4, 5, 6, 7], average="macro")
                # recall = tf_metrics.recall(label_ids, predictions, 11, [2, 3, 4, 5, 6, 7], average="macro")
                # f = tf_metrics.f1(label_ids, predictions, 11, [2, 3, 4, 5, 6, 7], average="macro")
                #
                return {
                    "eval_precision":precision,
                    "eval_recall":recall,
                    "eval_f": f,
                    #"eval_loss": loss,
                }
            eval_metrics = (metric_fn, [label_ids, pred_ids])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn
            )
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"predictions":pred_ids},   ### 是否要组织成一个dictionary?
                scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


# def load_data():
#     processer = NerProcessor()
#     processer.get_labels()
#     example = processer.get_train_examples(FLAGS.data_dir)
#     print()

def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        tf.logging.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last


def adam_filter(model_path):
    """
    去掉模型中的Adam相关参数，这些参数在测试的时候是没有用的
    :param model_path: 
    :return: 
    """
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    # to make sure the output_dir exists
    tf.gfile.MakeDirs(FLAGS.output_dir)

    processors = {
        "ner": NerProcessor
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))



    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case
    )

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project
        )

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list)+1,  # why? label id starts from 1
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples,
            label_list,
            FLAGS.max_seq_length,
            tokenizer,
            train_file)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            is_eval=False,
            drop_remainder=True)

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).

            # 8 is the TPU cores
            while len(eval_examples) % (FLAGS.eval_batch_size * FLAGS.num_tpu_cores) != 0:
                eval_examples.append(PaddingInputExample())


        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
            # The total batch size should be a multiple of 64 (8 per TPU core), and feature dimensions should be a multiple of 128
            # https://cloud.google.com/tpu/docs/troubleshooting
            # eval_steps = eval_steps // 8 * 8
            # solved by padding

        eval_drop_remainder = True if FLAGS.use_tpu else False
        #eval_drop_remainder = False

        if eval_steps is None:
            tf.logging.info("  eval_steps: None")
        else:
            tf.logging.info("  eval_steps = %d", eval_steps)

        tf.logging.info("  eval_drop_remainder = %d", int(eval_drop_remainder))

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            is_eval=True,
            drop_remainder=eval_drop_remainder)

        try:
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        except tf.errors.OutOfRangeError:
            tf.logging.info("Out Of Range error cached")

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        #token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        with tf.gfile.Open(os.path.join(FLAGS.output_dir, 'label2id.pkl'),'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value:key for key,value in label2id.items()}
        #if os.path.exists(token_path):
        #    os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file,mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            is_eval=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.txt")

        def result_to_pair(writer):
            for predict_line, prediction in zip(predict_examples, result):
                idx = 0
                line = ''
                line_token = str(predict_line.text).split(example_col_sep)
                label_token = str(predict_line.label).split(example_col_sep)
                if len(line_token) != len(label_token):
                    tf.logging.info(predict_line.text)
                    tf.logging.info(predict_line.label)
                for id in prediction:
                    if id == 0:
                        continue
                    curr_labels = id2label[id]
                    if curr_labels in ['[CLS]', '[SEP]']:
                        continue
                    # 不知道为什么，这里会出现idx out of range 的错误。。。do not know why here cache list out of range exception!
                    try:
                        line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                    except Exception as e:
                        tf.logging.info(e)
                        tf.logging.info(predict_line.text)
                        tf.logging.info(predict_line.label)
                        line = ''
                        break
                    idx += 1
                writer.write(line + '\n')

        with tf.gfile.GFile(output_predict_file, 'w') as writer:
            result_to_pair(writer)

        with tf.gfile.GFile(output_predict_file, 'r') as reader:
            eval_result = return_report(reader)
            print(eval_result)


        # # needs to confirm
        # with tf.gfile.GFile(output_predict_file,'w') as writer:
        #     for prediction in result:
        #         predict = prediction["predictions"]
        #         output_line = "\n".join(id2label[id] for id in predict if id!=0) + "\n"
        #         writer.write(output_line)

        # with tf.gfile.GFile(output_predict_file, "w") as writer:
        #     num_written_lines = 0
        #     tf.logging.info("***** Predict results *****")
        #     for (i, prediction) in enumerate(result):
        #         probabilities = prediction["probabilities"]
        #         if i >= num_actual_predict_examples:
        #             break
        #         output_line = "\t".join(
        #             str(class_probability)
        #             for class_probability in probabilities) + "\n"
        #         writer.write(output_line)
        #         num_written_lines += 1
        # assert num_written_lines == num_actual_predict_examples

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

# ===================
#     if args.do_train and args.do_eval:
#         # 加载训练数据
#         train_examples = processor.get_train_examples(args.data_dir)
#         num_train_steps = int(
#             len(train_examples) *1.0 / args.batch_size * args.num_train_epochs)
#         if num_train_steps < 1:
#             raise AttributeError('training data is so small...')
#         num_warmup_steps = int(num_train_steps * args.warmup_proportion)
#
#         tf.logging.info("***** Running training *****")
#         tf.logging.info("  Num examples = %d", len(train_examples))
#         tf.logging.info("  Batch size = %d", args.batch_size)
#         tf.logging.info("  Num steps = %d", num_train_steps)
#
#         eval_examples = processor.get_dev_examples(args.data_dir)
#
#         # 打印验证集数据信息
#         tf.logging.info("***** Running evaluation *****")
#         tf.logging.info("  Num examples = %d", len(eval_examples))
#         tf.logging.info("  Batch size = %d", args.batch_size)
#
#     label_list = processor.get_labels()
#     # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法，并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程
#     # tf 新的架构方法，通过定义model_fn 函数，定义模型，然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
#     # TPU不支持hook
#     model_fn = model_fn_builder(
#         bert_config=bert_config,
#         num_labels=len(label_list) + 1,
#         init_checkpoint=args.init_checkpoint,
#         learning_rate=args.learning_rate,
#         num_train_steps=num_train_steps,
#         num_warmup_steps=num_warmup_steps,
#         args=args)
#
#     params = {
#         'batch_size': args.batch_size
#     }
#
#     estimator = tf.estimator.Estimator(
#         model_fn,
#         params=params,
#         config=run_config)
#
#     if args.do_train and args.do_eval:
#         # 1. 将数据转化为tf_record 数据
#         train_file = os.path.join(args.output_dir, "train.tf_record")
#         if not os.path.exists(train_file):
#             filed_based_convert_examples_to_features(
#                 train_examples, label_list, args.max_seq_length, tokenizer, train_file, args.output_dir)
#
#         # 2.读取record 数据，组成batch
#         train_input_fn = file_based_input_fn_builder(
#             input_file=train_file,
#             seq_length=args.max_seq_length,
#             is_training=True,
#             drop_remainder=True)
#         # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
#
#         eval_file = os.path.join(args.output_dir, "eval.tf_record")
#         if not os.path.exists(eval_file):
#             filed_based_convert_examples_to_features(
#                 eval_examples, label_list, args.max_seq_length, tokenizer, eval_file, args.output_dir)
#
#         eval_input_fn = file_based_input_fn_builder(
#             input_file=eval_file,
#             seq_length=args.max_seq_length,
#             is_training=False,
#             drop_remainder=False)
#
#         # train and eval togither
#         # early stop hook
#         early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
#             estimator=estimator,
#             metric_name='loss',
#             max_steps_without_decrease=num_train_steps,
#             eval_dir=None,
#             min_steps=0,
#             run_every_secs=None,
#             run_every_steps=args.save_checkpoints_steps)
#
#         train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
#                                             hooks=[early_stopping_hook])
#         eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
#         tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
#
#     if args.do_predict:
#         token_path = os.path.join(args.output_dir, "token_test.txt")
#         if os.path.exists(token_path):
#             os.remove(token_path)
#
#         with codecs.open(os.path.join(args.output_dir, 'label2id.pkl'), 'rb') as rf:
#             label2id = pickle.load(rf)
#             id2label = {value: key for key, value in label2id.items()}
#
#         predict_examples = processor.get_test_examples(args.data_dir)
#         predict_file = os.path.join(args.output_dir, "predict.tf_record")
#         filed_based_convert_examples_to_features(predict_examples, label_list,
#                                                  args.max_seq_length, tokenizer,
#                                                  predict_file, args.output_dir, mode="test")
#
#         tf.logging.info("***** Running prediction*****")
#         tf.logging.info("  Num examples = %d", len(predict_examples))
#         tf.logging.info("  Batch size = %d", args.batch_size)
#
#         predict_drop_remainder = False
#         predict_input_fn = file_based_input_fn_builder(
#             input_file=predict_file,
#             seq_length=args.max_seq_length,
#             is_training=False,
#             drop_remainder=predict_drop_remainder)
#
#         result = estimator.predict(input_fn=predict_input_fn)
#         output_predict_file = os.path.join(args.output_dir, "label_test.txt")
#
#         def result_to_pair(writer):
#             for predict_line, prediction in zip(predict_examples, result):
#                 idx = 0
#                 line = ''
#                 line_token = str(predict_line.text).split(' ')
#                 label_token = str(predict_line.label).split(' ')
#                 len_seq = len(label_token)
#                 if len(line_token) != len(label_token):
#                     tf.logging.info(predict_line.text)
#                     tf.logging.info(predict_line.label)
#                     break
#                 for id in prediction:
#                     if idx >= len_seq:
#                         break
#                     if id == 0:
#                         continue
#                     curr_labels = id2label[id]
#                     if curr_labels in ['[CLS]', '[SEP]']:
#                         continue
#                     try:
#                         line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
#                     except Exception as e:
#                         tf.logging.info(e)
#                         tf.logging.info(predict_line.text)
#                         tf.logging.info(predict_line.label)
#                         line = ''
#                         break
#                     idx += 1
#                 writer.write(line + '\n')
#
#         with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
#             result_to_pair(writer)
#
#         eval_result = conlleval.return_report(output_predict_file)
#         print(''.join(eval_result))
#         # 写结果到文件中
#         with codecs.open(os.path.join(args.output_dir, 'predict_score.txt'), 'a', encoding='utf-8') as fd:
#             fd.write(''.join(eval_result))
#     # filter model
#     if args.filter_adam_var:
#         adam_filter(args.output_dir)

