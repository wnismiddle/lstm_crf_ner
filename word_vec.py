# -*- coding: utf-8 -*-
#构建词向量
import gensim.models #载入word2vec模块
from nltk.tokenize import TweetTokenizer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)#引入日志配置
def loaddata(inputfile):
    file = open(inputfile, 'r', encoding = 'utf-8')
    tknzr = TweetTokenizer()
    sentences=[]
    while 1:
        line = file.readline().strip()
        if not line:
              break
        sentences.append(tknzr.tokenize(line))
    return sentences


def trainVec(inputfile,outVectorFile):
    sentences=loaddata(inputfile)#加载分词后的文档
    modelbase = gensim.models.Word2Vec(min_count=1,size=100)#形成词向量模型
    modelbase.build_vocab(sentences)#构建词典树结构
    modelbase.wv.save_word2vec_format(outVectorFile)#存为词向量


if __name__=="__main__":#主函数
    inputfile="data/pre_word_vec.txt"#分词后的文件
    outVectorFile="wiki_100.utf8"#存成的词向量文件
    trainVec(inputfile,outVectorFile)

