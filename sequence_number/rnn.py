import tensorflow as tf
import numpy as np
class LSTM_RNN:

    def __init__(self,data,target,learning_rate):
        self.data = data
        self.target = target
        self.learning_rate = learning_rate
        self._prediction = None
        self._optimizer = None
        self._accurate = None

    @property
    def prediction(self):
        pass

    @property
    def optimizer(self):
        if not self._optimizer:
            cross_entropy = -tf.reduce_sum(self.target,tf.log(self.prediction))
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self._optimizer = optimizer.minimize(cross_entropy)
        return self._optimizer
    @property
    def accurate(self):
            accurate = tf.equal(tf.argmax(self.target,1),tf.arg_max(self.prediction,1))
            return tf.reduce_mean(tf.cast(accurate,tf.float32))