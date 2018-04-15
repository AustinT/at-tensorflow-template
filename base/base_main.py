import tensorflow as tf
import argparse
import os

from utils.config_reader import process_config
from utils.logger import Logger


class BaseMain:
    def __init__(self, model, data_gen, trainer):
        self.config = process_config(self.get_args().config)
        self.model = model
        self.data_gen = data_gen
        self.trainer = trainer
        self.make_dirs()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config',
                            metavar='C',
                            default='None',
                            help='The Configuration file')
        args = parser.parse_args()
        return args

    def make_dirs(self):
        for path in [self.config["summary_dir"], self.config["checkpoint_dir"]]:
            if not os.path.exists(path):
                os.makedirs(path)

    def main(self, test=False):
        with tf.Session() as sess:
            model = self.model(self.config)
            data_gen = self.data_gen(self.config)
            logger = Logger(sess, self.config)
            trainer = self.trainer(sess, model, data_gen, self.config, logger, load=True)
            trainer.train()

            if test:
                trainer.test_model()
