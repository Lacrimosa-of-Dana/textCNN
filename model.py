import tensorflow.compat.v1 as tf


class TextCNN:
    def __init__(self, class_num, sequence_size, vocabulary_size,
                 embed_size, embed_model, filter_sizes, filter_num) :
        self.class_num = class_num
        self.sequence_size = sequence_size

        self.input_x = tf.placeholder(tf.int32, shape=[None, self.sequence_size], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.class_num], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.vocabulary_size = vocabulary_size
        self.embed_size = embed_size

        self.iterator = tf.placeholder(tf.int32)
        # self.keep_prob = tf.placeholder(tf.float32)

        self.embed_model = tf.get_variable('embed_model', initializer=embed_model)
        self.embedded_words = None

        self.filter_sizes = filter_sizes
        self.filter_num = filter_num
        self.filter_total = len(self.filter_sizes) * self.filter_num
        self.l2_loss = tf.constant(0.0, dtype=tf.float32)
        self.h_out = None
        self.logits = None
        self.prediction = None
        self.loss = None
        self.accuracy = None

    def cnn(self):
        self.embedded_words = tf.expand_dims(
            tf.nn.embedding_lookup(self.embed_model, self.input_x), -1)
        pool_output = []
        for index, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.embed_size, 1, self.filter_num]
            filter_core = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1, dtype=tf.float64), dtype=tf.float64, name='filter_core')
            bias = tf.Variable(
                tf.constant(0.1, shape=[self.filter_num], dtype=tf.float32), name='bias')
            conv = tf.cast(tf.nn.conv2d(self.embedded_words, filter_core, strides=[1, 1, 1, 1],
                                padding='VALID', name='convolution'), tf.float32)
            h = tf.nn.relu(tf.nn.bias_add(conv, bias), name='relu')
            pool = tf.nn.max_pool(h, ksize=[1, self.sequence_size - filter_size + 1, 1, 1],
                                  strides=[1, 1, 1, 1], padding='VALID')
            pool_output.append(pool)
        self.h_out = tf.reshape(tf.concat(pool_output, 3), [-1, self.filter_total])
        drop = tf.nn.dropout(self.h_out, keep_prob=self.dropout_keep_prob)
        self.h_out = tf.layers.dense(drop, self.filter_total, activation=tf.nn.tanh, use_bias=True)
        W = tf.get_variable('W', initializer=tf.glorot_uniform_initializer(),
                            shape=[self.filter_total, self.class_num])
        b = tf.Variable(tf.constant(0.1, shape=[self.class_num], dtype=tf.float32), name='b')
        self.l2_loss += tf.nn.l2_loss(b)
        self.logits = tf.nn.xw_plus_b(drop, W, b, name='scores')
        self.prediction = tf.argmax(self.logits, 1, name='predictions')

    def cal_loss(self, l2_reg_lambda=0.01):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(loss) + l2_reg_lambda * self.l2_loss

    def cal_accuracy(self):
        correct = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name='accuracy')
