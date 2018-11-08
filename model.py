# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn
from utils import result_to_json, result_to_json_simple
from data_utils import create_input, iobes_iob


class Model(object):
    def __init__(self, config):

        self.config = config
        self.lr = config["lr"] #学习率
        self.char_dim = config["char_dim"]  # 输入维度
        self.lstm_dim = config["lstm_dim"] # lstm层隐藏单元个数
        self.seg_dim = config["seg_dim"]    # Embedding size for segmentation, 0 if not used

        self.pos_dim = config["pos_dim"]
        self.num_poss = config["num_poss"]

        self.num_tags = config["num_tags"]  # 输出标签个数
        self.num_chars = config["num_chars"]    # 输入字符个数
        self.num_segs = 7

        self.global_step = tf.Variable(0, trainable=False) # 代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表（参考网络说法）。
        self.current_epoch = tf.Variable(0, trainable=False) # 代表全局迭代轮次
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)    # 验证集模型的最佳F值
        self.best_test_f1 = tf.Variable(0.0, trainable=False)   # 测试集模型的最佳F值
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model

        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.pos_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="PosInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")
        """
        referencec : https://www.tensorflow.org/api_docs/python/tf/shape
        y = tf.abs(x) the presentation of x = a+bj , y = 根号下的（a方+b方）
        ex：
            x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
            tf.abs(x)  # [5.25594902, 6.60492229]
        y = tf.sign(x) if x<0, y =-1 ;if x=0, y=0; if x>0, y=1
        y = reduce_sum(x) 计算输入tensor元素的和，或者按照reduction_indices指定的轴求和
        ex :
            x is [[1,1,1],[1,1,1]]
            tf.reduce_sum(x) = 6
            tf.reduce_sum(x, 0) = [2,2,2] 纵向相加
            tf.reduce_sum(x, 1) = [3,3] 横向相加
            tf.reduce_sum(x, 1, keep_dims = true) = [[3],[3]]
            tf.reduce_sum(x, [0,1]) = 6
        y = tf.cast(x) 改变x的type
        ex：
            x = tf.constant([1.8, 2.2], dtype=tf.float32)
            tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
        y = tf.shape(x) 返回x的shape
        ex：
            t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
            tf.shape(t)  # [2, 2, 3]
        """
        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1) # 压缩矩阵
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        # embeddings for chinese character and segmentation representation
        # embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, self.pos_inputs, config)

        # apply dropout before feed to lstm layer
        # tf.nn.dropout(x, keep_prob, noise_shape, seed, name)
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)

        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        # logits for tags
        self.logits = self.project_layer(lstm_outputs)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        # max_to_keep 表示要保留的最近检查点文件的最大数量。默认为5
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
    #     """
    #     :param char_inputs: one-hot encoding of sentence
    #     :param seg_inputs: segmentation feature
    #     :param config: wither use segmentation feature
    #     :return: [1, num_steps, embedding size],
    #     """
    #
    #     embedding = []
    #     # variable_scope对象携带get_variable对象 name: 当前范围的变量名
    #     with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
    #         self.char_lookup = tf.get_variable(
    #                 name="char_embedding",
    #                 shape=[self.num_chars, self.char_dim],
    #                 initializer=self.initializer)
    #         embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
    #         if config["seg_dim"]:
    #             with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
    #                 self.seg_lookup = tf.get_variable(
    #                     name="seg_embedding",
    #                     shape=[self.num_segs, self.seg_dim],
    #                     initializer=self.initializer)
    #                 # tf.nn.embedding_lookup(params, ids, partition_strategy, name, validate_indices, max_norm )
    #                 # params: A single tensor representing the complete embedding tensor
    #                 # ids: A Tensor with type int32 or int64 containing the ids to be looked up in params
    #                 # 返回A Tensor with the same type as the tensors in params
    #                 embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
    #         # tf.concat(values, axis, name) 按某个维度联结矩阵
    #         embed = tf.concat(embedding, axis=-1)
    #     # return embed

    def embedding_layer(self, char_inputs, seg_inputs, pos_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """

        embedding = []
        # variable_scope对象携带get_variable对象 name: 当前范围的变量名
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    # tf.nn.embedding_lookup(params, ids, partition_strategy, name, validate_indices, max_norm )
                    # params: A single tensor representing the complete embedding tensor
                    # ids: A Tensor with type int32 or int64 containing the ids to be looked up in params
                    # 返回A Tensor with the same type as the tensors in params
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
                if config["pos_dim"]:
                    with tf.variable_scope("pos_embedding"), tf.device('/cpu:0'):
                        self.pos_lookup = tf.get_variable(
                            name="pos_embedding",
                            shape=[self.num_poss, self.pos_dim],
                            initializer=self.initializer)
                        embedding.append(tf.nn.embedding_lookup(self.pos_lookup, pos_inputs))
            # tf.concat(values, axis, name) 按某个维度联结矩阵
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])
                pred = tf.nn.xw_plus_b(output, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])


    # def project_layer(self, lstm_outputs, name=None):
    #     """
    #     hidden layer between lstm layer and logits
    #     :param lstm_outputs: [batch_size, num_steps, emb_size]
    #     :return: [batch_size, num_steps, num_tags]
    #     """
    #     with tf.variable_scope("project"  if not name else name):
    #         with tf.variable_scope("hidden"):
    #             W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
    #                                 dtype=tf.float32, initializer=self.initializer)
    #
    #             b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
    #                                 initializer=tf.zeros_initializer())
    #             output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
    #             hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
    #
    #         # project to score of tags
    #         with tf.variable_scope("logits"):
    #             W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
    #                                 dtype=tf.float32, initializer=self.initializer)
    #
    #             b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
    #                                 initializer=tf.zeros_initializer())
    #
    #             pred = tf.nn.xw_plus_b(hidden, W, b)
    #
    #         return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        # _, chars, segs, tags = batch
        # feed_dict = {
        #     self.char_inputs: np.asarray(chars),
        #     self.seg_inputs: np.asarray(segs),
        #     self.dropout: 1.0,
        # }

        _, chars, segs, poss, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.pos_inputs: np.asarray(poss),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        # return result_to_json(inputs[0][0], tags)   # BIOES模式
        return result_to_json_simple(inputs[0][0], tags)    # BIO模式


