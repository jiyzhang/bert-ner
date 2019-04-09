#coding: utf-8

import numpy as np
from pathlib import Path
import os
from data import fwords, ftags, DATADIR, build_hanzi_vocab, build_tag_vocab, build_hanzi_embedding

path = DATADIR / 'boson_ann_NONB'
files = os.listdir(path)
files_txt = [i for i in files if i.endswith('.txt')]
files = [i[0:-4] for i in files_txt]

sent_file = DATADIR / "sent.txt"
tag_file  = DATADIR / "tag.txt"


def append_sents_tags(sent_f, tag_f, file):
    """
    读取brat annotation tool的标注结果，转换为如下格式：
    句子: 以"。"为分隔符，一个句子一行。句子内的字之间以"|"来分割
    标签：对应于每一个句子的标签，长度与句子相同，不同字的标签之间通过"|"分割
    :param sent_f: 输出句子文件的 file handler
    :param tag_f:  输出标签文件的file handler
    :param file:  文件名，不含扩展名。用于拼接处句子文件名(.txt)和标签文件名(.ann)
    :return:
    """
    sent_file = path / (file + ".txt")
    tag_file  = path / (file + ".ann")

    print(file)

    with Path(sent_file).open() as f_s, Path(tag_file).open() as f_t:
        # 公告的全文，只有一行
        sent_content = f_s.readline().strip()
        sent_size = len(sent_content)
        tags_content = list(['O'] * sent_size)

        tag_lines = f_t.readlines()
        for tag_line in tag_lines:
            alist = tag_line.split()
            if len(alist) > 5:
                #忽略英文公司名称
                #new_008739b9378aa74c793113e7335f6638
                """
T28     COMPANY 727 746 Golden Ares Limited
T29     COMPANY 747 768 Gingko Avenue Limited
T30     COMPANY 769 795 Mission Excellence Limited
T31     PERSON 796 799  王培强
T32     COMPANY 800 815 福建平潭自贸区崇德投资合伙企业
T33     COMPANY 822 839 Sagacious Limited
T34     PERSON 840 842  刘承
T35     PERSON 844 847  周小童
T38     COMPANY 1006 1016       神州优车股份有限公司
T1      COMPANY 410 414 中国结算
T2      COMPANY 700 726 Star Vantage(China)Limited
                """
                continue

            (_, tag, start, end, entity) = tag_line.split()

            #print("{},{}, total size: {}".format(start, end, sent_size))
            start = int(start)
            end = int(end)
            # print(line_arr[end])
            # print(line_arr[start:end] )
            if tag == "PERSON":
                tags_content[start] = "B-PER"  # B-PER
                for i in range(start + 1, end):
                    tags_content[i] = "I-PER"
                #tags_content[start + 1: end] = "I-PER"  # I-PER
            if tag == "COMPANY":
                tags_content[start] = "B-ORG"  # B-ORG
                for i in range(start + 1, end):
                    tags_content[i] = "I-ORG"
                #tags_content[start + 1: end] = "I-ORG"  # B-ORG

        #按照"。"分成句子

        sent_arr = sent_content.split("。")

        sent_len = 0
        sent_acc = 0

        startpos = 0
        for sentence in sent_arr:
            sent_len = len(sentence)
            sent_str_to_write = "|".join(list(sentence))

            sent_f.write(sent_str_to_write)
            sent_f.write("\n")

            tag_str_to_write = "|".join(tags_content[startpos: startpos + sent_len])
            tag_f.write(tag_str_to_write)
            tag_f.write("\n")

            # print("sent_len: {}, tag_len: {}".format(len(sent_splitted.split()), len(tag_str.split())))
            # if len(sent_splitted.split()) != len(tag_str.split()):
            #     print(sent_splitted)
            #     print(tag_str)
            ## split会出现不相等的情况，因为sent中会有多个空格在一起。
            #assert len(sent_splitted.split()) == len(tag_str.split()), "the length of sent and tag don't match"
            assert(len(sentence) == len(tags_content[startpos: startpos + sent_len]))
            startpos = startpos + sent_len + 1 # skip the "。"

if __name__ == '__main__':
    # 生成sent.txt, tag.txt
    with Path(sent_file).open("w") as sent_f, Path(tag_file).open("w") as tag_f:
        for i in files:
            append_sents_tags(sent_f, tag_f, i)


    #将sent.txt, tag.txt的内容切分train, valid, test
    #生成文件名:
    #test.tags.txt
    #test.words.txt
    #train.tags.txt
    #train.words.txt
    #valid.tags.txt
    #valid.words.txt
    with Path(sent_file).open("r") as sent_f, Path(tag_file).open("r") as tag_f:
        sent_lines = sent_f.readlines()
        tag_lines  = tag_f.readlines()

        total_size = len(sent_lines)
        train_size = total_size * 0.7
        valid_size = total_size * 0.2
        test_size  = total_size - train_size - valid_size

        i = 0

        f_train_words = Path(fwords("train")).open("w")
        f_train_tags  = Path(ftags("train")).open("w")

        f_valid_words = Path(fwords("valid")).open("w")
        f_valid_tags  = Path(ftags("valid")).open("w")

        f_test_words = Path(fwords("test")).open("w")
        f_test_tags  = Path(ftags("test")).open("w")

        for s, t in zip(sent_lines, tag_lines):
            if len(s.strip()) != 0:
                if i < train_size:
                    f_train_words.write(s)
                    #f_train_words.write("\n")
                    f_train_tags.write(t)
                    #f_train_tags.write("\n")
                elif i < train_size + valid_size:
                    f_valid_words.write(s)
                    #f_valid_words.write("\n")
                    f_valid_tags.write(t)
                    #f_valid_tags.write("\n")
                else:
                    f_test_words.write(s)
                    #f_test_words.write("\n")
                    f_test_tags.write(t)
                    #f_test_tags.write("\n")

            i = i + 1

        f_train_words.close()
        f_train_tags.close()
        f_valid_words.close()
        f_valid_tags.close()
        f_test_words.close()
        f_test_tags.close()

    # 生成汉字字表和标记字表
    # 生成文件 vocab.words.txt
    build_hanzi_vocab()
    # 生成文件 vocab.tags.txt
    build_tag_vocab()

    # 根据中文字词向量表，生成对应汉字字表的embedding
    # sgns.npz
    build_hanzi_embedding()

