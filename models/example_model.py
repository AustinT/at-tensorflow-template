from base.base_model import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config["input_size"])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])

        # network architecture
        d1 = tf.layers.dense(self.x, 512, activation=tf.nn.elu, name="dense1")
        d2 = tf.layers.dense(d1, 1, name="dense2")
        self.preds = d2

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.square(d2 - self.y))
            optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])
            self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])

