from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score


# 第六列数据字段名称设为“梯形面积”，以S2为例，S2=1/2*（收入百分比1+收入百分比2）*（人数百分比2-人数百分比1）
# 按照前述的计算原理编辑公式即可，通过以上步骤，最后将“梯形面积”字段的数值累加，即为前文所述的B面积
# 没太看懂
def gini(actual, pred):
    assert (len(actual) == len(pred))  # 行数必须相等
    # 三列数据，实际值，预测值，自然数增长
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]  # 按第1列从大到小排序   因为-1×第1列
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


class DeepFm:
    def __init__(self, feature_size, field_size, embedding_size=8, dropout_fm=[1.0, 1.0], deep_layers=[32, 32],
                 dropout_deep=[0.5, 0.5, 0.5], deep_layer_activation=tf.nn.relu, epoch=30, batch_size=1024,
                 learning_rate=0.01, optimizer='adam', batch_norm=0, batch_norm_decay=0.995, verbose=False,
                 random_seed=2019, use_fm=True, use_deep=True, loss_type='logloss', eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.dropout_fm = dropout_fm
        self.dropout_deep = dropout_deep
        self.deep_layers = deep_layers
        self.deep_layers_activation = deep_layer_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')
            self.label = tf.placeholder(tf.int32, shape=[None, 1], name='label')
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')
            self.train_phase = tf.placeholder(bool, name='train_phase')

            # 由函数生成
            self.weights = self._initialize_weights()

            # model    TMD  multiply这个函数是普通相乘，各自元素相乘，matmul才是矩阵相乘
            # N*F*K
            self.embedding = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embedding = tf.multiply(self.embedding, feat_value)

            #  fm first order
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
            # 此处的dropout_keep_fm 为什么取第0个
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])

            # fm second order term
            # sum-square-part
            self.summed_feature_emb = tf.reduce_sum(self.embedding, 1)
            self.summed_feature_emb_square = tf.square(self.summed_feature_emb)

            # square-sum-part
            self.square_feature_emb = tf.square(self.embedding)
            self.square_feature_emb_summed = tf.reduce_sum(self.square_feature_emb, 1)

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_feature_emb_square, self.square_feature_emb_summed)
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])

            # deep component
            self.y_deep = tf.reshape(self.embedding, [-1, self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            for i in range(len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights['layer_{}'.format(i)]), self.weights['bias_{}'.format(i)])
                self.y_deep = self.deep_layers_activation(self.y_deep)  # tf.nn.relu
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])

            if self.use_deep and self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            else:
                concat_input = self.y_deep

            self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])

            # loss
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['concat_projection'])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['layer_{}'.format(i)])

            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, ).minimize(
                    self.loss)
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.5).minimize(
                    self.loss)

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        weights = dict()
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 1.0), name='feature_bias')

        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size  # 输入的数据数，  矩阵相乘时  [3,4]*[4,5]=[3,5]
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))  # 不知道这个为什么

        weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])),
                                         dtype=np.float32)
        weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                        dtype=np.float32)
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])), dtype=np.float32)  # 1 * layer[i]

        # 这个值得好好看下
        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                                                   dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)
        return weights

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None, y_valid=None, early_stopping=False,
            refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            # 打乱数据
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            # 这个我觉得应该+1 有小数的时候 int直接舍去了
            iteration = int(len(y_train) / self.batch_size)
            for i in range(iteration):
                # mini batch 分次训练
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            # 跑完一整个数据集，验证一下结果怎么样
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            # 追加保存每一次的结果
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print('epoch:%d, train_result:%.4f, valid_result:%.4f, time:%.1f'
                          % (epoch + 1, train_result, valid_result, time()-t1))
                else:
                    print('epoch:%d, train_result:%.4f, time:%.1f' % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.train_termination(self.valid_result):
                break

    def shuffle_in_unison_scary(self, a, b, c):  # 同样的顺序打乱
        # shuffle three lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)  # 设置同样的打乱顺序
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]  # y就成了[[1],[7],[6]]这种形式

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    # 返回评价指标，一般是越大越好
    def evaluate(self, Xi, Xv, y):
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)

    def predict(self, Xi, Xv):
        # dummy y in order to get the same of Xi.shape
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred

    # 这个为什么 加入到跳出循环的条件
    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] < valid_result[-3] < valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] > valid_result[-3] > valid_result[-4] > valid_result[-5]:
                    return True
        return False
