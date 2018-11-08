# -*- coding:utf-8 -*-

import codecs
import random
import sys
#设置最大递归深度
sys.setrecursionlimit(1000000)

global note_dic
global dic
global max_chars
read_path = 'data/train_data600/'
write_path = 'data/'
tag2label = {
    "解剖部位": 'par',
    "手术": 'sur',
    "症状描述": 'des',
    "药物": 'drug',
    "独立症状": 'sys'
}

def combine_txt():
    write_file = write_path + 'origin/combine_sum.txt'
    f_write = open(write_file, 'w', encoding='utf-8')
    for i in range(1, 601):
        file = read_path + '入院记录现病史-' + str(i) + '.txtoriginal.txt'
        f_org = open(file, 'r', encoding='utf-8')
        f_write.write(''.join(f_org.readlines()))
        f_org.close()
    f_write.close()

def combine_note():
    write_file = write_path + 'origin/combine_note.txt'
    f_write = open(write_file, 'w', encoding='utf-8')
    for i in range(1, 601):
        file = read_path + '入院记录现病史-' + str(i) + '.txt'
        f_org = open(file, 'r', encoding='utf-8')
        f_write.write(''.join(f_org.readlines()))
        f_org.close()
    f_write.close()

def split_by_note():
    # 按标注分词，结果汇总写入txt，每个样本为一行
    write_file = write_path + 'origin/note_sum.txt'
    f_sum = open(write_file, 'w', encoding='utf-8')

    # 遍历原始txt和标注txt，按标注进行分词
    for i in range(1, 601):
        org_file = read_path + '入院记录现病史-' + str(i) + '.txtoriginal.txt'
        note_file = read_path + '入院记录现病史-' + str(i) + '.txt'
        f_org = open(org_file, 'r', encoding='utf-8')
        f_note = open(note_file, 'r', encoding='utf-8')
        index = 0
        content = ''.join(f_org.readlines())    # 原始病历
        content_new = ''                        # 分词病历，按“/”拼接
        # 读取标注txt，并分词
        for line in f_note.readlines():
            list = line.split("	")
            if index < int(list[1]):
                content_new += content[index:int(list[1])]+'/o '
                content_new += content[int(list[1]):int(list[2])]+'/'+tag2label.get(list[3].strip())+' '
                index = int(list[2])
        # 分词病历写入汇总txt
        f_sum.write(content_new + '\n')
        f_org.close()
        f_note.close()

    f_sum.close()
    print('按标注分词完成！')

def split_char():
    # 按标注分词，结果汇总写入txt，每个样本为一行
    write_file = write_path + 'origin/char_sum.txt'
    f_sum = open(write_file, 'w', encoding='utf-8')

    # 遍历原始txt和标注txt，按标注进行分词
    for i in range(1, 601):
        org_file = read_path + '入院记录现病史-' + str(i) + '.txtoriginal.txt'
        note_file = read_path + '入院记录现病史-' + str(i) + '.txt'
        split_file = write_path + 'splitword/入院记录现病史-splitword-' + str(i) + '.txt'
        f_org = open(org_file, 'r', encoding='utf-8')
        f_note = open(note_file, 'r', encoding='utf-8')
        f_split = open(split_file, 'w', encoding='utf-8')

        index = 0
        content = ''.join(f_org.readlines())  # 原始病历
        content_new = ''

        # 读取标注txt，并分词
        for line in f_note.readlines():
            list = line.split("	")
            note = tag2label.get(str(list[3]).strip())
            while index < int(list[1]):
                content_new += content[index:index+1] + '\tO\n'
                index +=1
            content_new += content[index:index+1] + '\tB-' + note + '\n'
            index += 1
            while index<int(list[2]):
                content_new += content[index:index + 1] + '\tI-' + note + '\n'
                index += 1

        f_split.write(content_new)
        f_split.close()

        # 分词病历写入汇总txt
        f_sum.write(content_new + '\n')
        f_org.close()
        f_note.close()

    f_sum.close()

def split_char_by_rate(train_rate = 0.7, dev_rate = 0.3):
    f_train = open(write_path + 'example.train', 'w', encoding='utf-8')
    f_dev = open(write_path + 'example.dev', 'w', encoding='utf-8')
    f_test = open(write_path + 'example.test', 'w', encoding='utf-8')

    for i in range(1, 601):
        if (i < int(600 * train_rate)) :
            f_write = f_train
        elif (i<int(600 * (train_rate + dev_rate))) :
            f_write = f_dev
        else :
            f_write = f_test
        f_org = open(read_path + '入院记录现病史-' + str(i) + '.txtoriginal.txt', 'r', encoding='utf-8')
        f_note = open(read_path + '入院记录现病史-' + str(i) + '.txt', 'r', encoding='utf-8')
        f_split = open(write_path + 'splitword/入院记录现病史-splitword-' + str(i) + '.txt', 'w', encoding='utf-8')

        index = 0
        content = ''.join(f_org.readlines())  # 原始病历
        content_new = ''

        # 读取标注txt，并分词
        for line in f_note.readlines():
            list = line.split("	")
            note = tag2label.get(str(list[3]).strip())
            while index < int(list[1]):
                word = content[index:index+1]
                if not word.strip():
                    word = "@"
                content_new += word + ' O\n'
                index +=1
            word = content[index:index + 1]
            if not word.strip():
                word = "@"
            content_new += word + ' B-' + note + '\n'
            index += 1
            while index<int(list[2]):
                word = content[index:index+1]
                if not word.strip():
                    # print(index)
                    # print('*****{}*****{}***'.format(i, word))
                    # print(content[index-1:index])
                    # print(line)
                    word = "@"
                content_new += word + ' I-' + note + '\n'
                index += 1
        f_split.write(content_new)
        f_split.close()

        # 分词病历写入汇总txt
        f_write.write(content_new + '\n')
        f_org.close()
        f_note.close()
    f_train.close()
    f_dev.close()
    f_test.close()

def split_entity(train_rate = 0.7, dev_rate = 0.3):
    f_train = open(write_path + 'ccks_train', 'w', encoding='utf-8')
    f_dev = open(write_path + 'ccks_dev', 'w', encoding='utf-8')

    for i in range(1, 601):
        if (i < int(600 * train_rate)) :
            f_write = f_train
        else:
            f_write = f_dev
        f_org = open(read_path + '入院记录现病史-' + str(i) + '.txtoriginal.txt', 'r', encoding='utf-8')
        f_note = open(read_path + '入院记录现病史-' + str(i) + '.txt', 'r', encoding='utf-8')
        index = 0
        content = ''.join(f_org.readlines())  # 原始病历
        content_new = ''
        # 读取标注txt，并分词
        for line in f_note.readlines():
            list = line.split("	")
            note = tag2label.get(str(list[3]).strip())
            while index < int(list[1]):
                word = content[index:index+1]
                if not word.strip():
                    word = "@"
                content_new += word + ' O\n'
                index +=1
            word = content[index:index + 1]
            if not word.strip():
                word = "@"
            content_new += word
            index += 1
            while index<int(list[2]):
                word = content[index:index+1]
                if not word.strip():
                    word = "@"
                content_new += word
                index += 1
            content_new += ' ' + note + '\n'

        # 分词病历写入汇总txt
        f_write.write(content_new + '\n')
        f_org.close()
        f_note.close()
    f_train.close()
    f_dev.close()

def generate_samples_by_rate(rate = 0.7):
    # 划分数据集
    f_train = open(write_path + 'train_src.txt', 'w', encoding='utf-8')
    f_tgt = open(write_path + 'train_tgt.txt', 'w', encoding='utf-8')
    f_test = open(write_path + 'test_src.txt', 'w', encoding='utf-8')
    f_test_tgt = open(write_path + 'test_tgt.txt', 'w', encoding='utf-8')

    # 遍历原始txt和标注txt，按标注进行分词
    train_num = int(600 * rate)
    for i in range(1, 601):
        # 训练集-测试集 按比例（默认7/3）划分
        if (i < train_num):
            f_src= f_train
            f_tgt = f_tgt
        else:
            f_src= f_test
            f_tgt = f_test_tgt
        f_org = open(read_path + '入院记录现病史-' + str(i) + '.txtoriginal.txt', 'r', encoding='utf-8')
        f_note = open(read_path + '入院记录现病史-' + str(i) + '.txt', 'r', encoding='utf-8')
        f_split = open(write_path + 'splitword/入院记录现病史-splitword-' + str(i) + '.txt', 'w', encoding='utf-8')

        index = 0
        content = ''.join(f_org.readlines())  # 原始病历
        content_new = ''
        note_new = ''

        # 读取标注txt，并分词
        for line in f_note.readlines():
            list = line.split("	")
            note = tag2label.get(str(list[3]).strip())
            while index < int(list[1]):
                content_new += content[index:index + 1] + '\t'
                note_new += 'O' + '\t'
                index += 1
            content_new += content[index:index + 1] + '\t'
            note_new += 'B-' + note + '\t'
            index += 1
            while index < int(list[2]):
                content_new += content[index:index + 1] + '\t'
                note_new += 'I-' + note + '\t'
                index += 1
        f_split.write(content_new)
        f_split.close()
        f_org.close()
        f_note.close()

        # 分词病历写入汇总txt
        f_src.write(content_new + '\n')
        f_tgt.write(note_new + '\n')

    f_test_tgt.close()
    f_test.close()
    f_tgt.close()
    f_train.close()

def pre_word_vec():
    # 合并1000个文本
    f_word = open('data/pre_word_vec.txt', 'w', encoding='utf-8')
    for i in range(1, 601):
        f_org = open(read_path + '入院记录现病史-' + str(i) + '.txtoriginal.txt', 'r', encoding='utf-8')
        content = ''.join(f_org.readlines())
        content_new = []
        for index in range(len(content)):
            word = content[index:index + 1]
            if not word.strip():
                word = "@"
            content_new.append(word)
        f_word.write(" ".join(content_new)+'\n')
    for i in range(1, 401):
        f_org = open('data/testdata/入院记录现病史-' + str(i) + '.txtoriginal.txt', 'r', encoding='utf-8')
        content = ''.join(f_org.readlines())
        content_new = []
        for index in range(len(content)):
            word = content[index:index + 1]
            if not word.strip():
                word = "@"
            content_new.append(word)
        f_word.write(" ".join(content_new)+'\n')
    f_word.close()

def random_samples(num = 600, train_rate = 0.8, dev_rate = 0.2):
    # 训练集、验证集、测试集随机取样：打乱下标，分别存入三个集合中
    total_numset = set([x for x in range(1,601)])
    train_numset = set(random.sample(total_numset, int(train_rate*num)))
    rest = total_numset.difference(train_numset)
    dev_numset = set(random.sample(rest, int(dev_rate*num)))
    rest = total_numset.difference(train_numset|dev_numset)
    test_numset = rest

    assert len(train_numset|dev_numset|test_numset) == num      # 保证训练集、验证集、测试集数量总和为总样本数

    f_train = open(write_path + 'example.train', 'w', encoding='utf-8')
    f_dev = open(write_path + 'example.dev', 'w', encoding='utf-8')
    f_test = open(write_path + 'example.test', 'w', encoding='utf-8')

    def generate_data(numset, f_write):
        f_write = f_write
        for i in numset:
            f_org = open(read_path + '入院记录现病史-' + str(i) + '.txtoriginal.txt', 'r', encoding='utf-8')
            f_note = open(read_path + '入院记录现病史-' + str(i) + '.txt', 'r', encoding='utf-8')
            f_split = open(write_path + 'splitword/入院记录现病史-splitword-' + str(i) + '.txt', 'w', encoding='utf-8')
            index = 0
            content = ''.join(f_org.readlines())  # 原始病历
            content_new = ''
            # 读取标注txt，并分词
            for line in f_note.readlines():
                list = line.split("	")
                note = tag2label.get(str(list[3]).strip())
                while index < int(list[1]):
                    word = content[index:index + 1]
                    if not word.strip():
                        word = "@"
                    content_new += word + ' O\n'
                    index += 1
                word = content[index:index + 1]
                if not word.strip():
                    word = "@"
                content_new += word + ' B-' + note + '\n'
                index += 1
                while index < int(list[2]):
                    word = content[index:index + 1]
                    if not word.strip():
                        # print(index)
                        # print('*****{}*****{}***'.format(i, word))
                        # print(content[index-1:index])
                        # print(line)
                        word = "@"
                    content_new += word + ' I-' + note + '\n'
                    index += 1
            f_split.write(content_new)
            f_split.close()
            # 分词病历写入汇总txt
            f_write.write(content_new + '\n')
            f_org.close()
            f_note.close()

    generate_data(train_numset, f_train)
    generate_data(dev_numset, f_dev)
    generate_data(test_numset, f_test)
    f_train.close()
    f_dev.close()
    f_test.close()

# 总样本，分K份，第pos份作为测试集，其余做训练集
def K_pieces(num = 600, k = 5, pos = 5):
    batch = num/k
    f_train = open(write_path + 'example.train', 'w', encoding='utf-8')
    f_dev = open(write_path + 'example.dev', 'w', encoding='utf-8')

    for i in range(1, 601):
        if i > (pos-1)*batch and i <= pos *batch:
            f_write = f_dev
            print('测试集下标', i)
        else:
            print('---训练集下标', i)
            f_write = f_train
        f_org = open(read_path + '入院记录现病史-' + str(i) + '.txtoriginal.txt', 'r', encoding='utf-8')
        f_note = open(read_path + '入院记录现病史-' + str(i) + '.txt', 'r', encoding='utf-8')
        index = 0
        content = ''.join(f_org.readlines())  # 原始病历
        content_new = ''
        # 读取标注txt，并分词
        for line in f_note.readlines():
            list = line.split("	")
            note = tag2label.get(str(list[3]).strip())
            while index < int(list[1]):
                word = content[index:index + 1]
                if not word.strip():
                    word = "@"
                content_new += word + ' O\n'
                index += 1
            word = content[index:index + 1]
            if not word.strip():
                word = "@"
            content_new += word + ' B-' + note + '\n'
            index += 1
            while index < int(list[2]):
                word = content[index:index + 1]
                if not word.strip():
                    word = "@"
                content_new += word + ' I-' + note + '\n'
                index += 1
        # 分词病历写入汇总txt
        f_write.write(content_new + '\n')
        f_org.close()
        f_note.close()
    f_train.close()
    f_dev.close()

def generate_test_samples():
    f_result = open('data/ccks1_result400t.txt', 'r', encoding='utf-8')
    f_test = open(write_path + 'example.test', 'w', encoding='utf-8')
    for line in f_result.readlines():
        order = line.split(',')[0]
        f_org = open('data/testdata/入院记录现病史-' + str(order) + '.txtoriginal.txt', 'r', encoding='utf-8')

        entities = line.split(',')[1].split(';')
        entities.pop()

        content = ''.join(f_org.readlines())  # 原始病历
        content_new = ''
        index = 0
        for entity in entities:
            list = entity.split('\t')
            note = tag2label.get(list[3].strip())
            while index < int(list[1]):
                word = content[index:index + 1]
                if not word.strip():
                    word = "@"
                content_new += word + ' O\n'
                index += 1
            word = content[index:index + 1]
            if not word.strip():
                word = "@"
            content_new += word + ' B-' + note + '\n'
            index += 1
            while index < int(list[2]):
                word = content[index:index + 1]
                if not word.strip():
                    word = "@"
                content_new += word + ' I-' + note + '\n'
                index += 1
        # 分词病历写入汇总txt
        f_test.write(content_new + '\n')
        f_org.close()
    f_test.close()
    f_result.close()

if __name__ == '__main__':
    # 合并600个病历
    # combine_txt()
    # combine_note()
    # split_char_by_rate()
    # split_char()
    # split_entity()
    # generate_samples_by_rate(0.7)
    random_samples()
    # pre_word_vec()
    # K_pieces()
    generate_test_samples()

