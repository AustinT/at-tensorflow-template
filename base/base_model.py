"""
Base model
"""
from sklearn.base import BaseEstimator
import tensorflow as tf


# TODO: add estimator methods!!!
class BaseModel(BaseEstimator):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_global_step()
        self.init_current_epoch()

    def init_global_step(self):
        with tf.variable_scope("global_step"):
            self.global_step = tf.Variable(0, trainable=False, name="global_step")

    def init_current_epoch(self):
        with tf.variable_scope("current_epoch"):
            self.current_epoch = tf.Variable(0, trainable=False, name="current_epoch")
            self.inc_curr_epoch = tf.assign_add(self.current_epoch, 1)

    def save(self, sess):
        self.saver.save(sess, self.config["checkpoint_full_path"], self.global_step)

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config["checkpoint_dir"])
        if latest_checkpoint:
            print("Loading checkpoint: {}".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)

    def init_saver(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError