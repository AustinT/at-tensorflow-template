import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, data, config, logger, load=True):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        # Load the model after initialization to ensure that the loaded values are kept
        if load:
            self.model.load(self.sess)

    def train(self):
        for cur_epoch in range(self.model.current_epoch.eval(self.sess), self.config["num_epochs"]):
            print("Start Epoch: {}".format(cur_epoch), flush=True)
            self.train_epoch()
            self.sess.run(self.model.inc_curr_epoch)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def test_model(self):
        raise NotImplementedError