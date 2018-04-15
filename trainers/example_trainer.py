from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger, **kwargs):
        super().__init__(sess, model, data, config,logger, **kwargs)

    def train_epoch(self):
        loop = tqdm(range(self.data.get_num_data() // self.config["batch_size"]))
        losses = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)

        cur_it = self.model.global_step.eval(self.sess)
        summaries_dict = {
            'loss': loss,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config["batch_size"]))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_step, self.model.loss],
                                     feed_dict=feed_dict)
        return loss

    def test_model(self):

        # Ideally this wouldn't be hard coded here, but it is just a test so whatever
        x = np.linspace(-1, 1, 200).reshape(-1, 1)
        y = self.sess.run(self.model.preds, feed_dict={self.model.x: x})
        plt.plot(x, x**2, label="Truth")
        plt.plot(x, y, label="pred")
        plt.legend()
        plt.show()
