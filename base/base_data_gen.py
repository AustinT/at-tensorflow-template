import numpy as np


class BaseDataGen:
    def __init__(self, config):
        self.config = config
        self.current_pos = 0
        self.data = self.load_data()

    def load_data(self):
        raise NotImplementedError

    def shuffle_data(self):
        raise NotImplementedError

    def get_num_data(self):
        raise NotImplementedError

    def random_batch(self, batch_size):
        indices = np.random.choice(self.get_num_data(), batch_size)
        yield (data[indices] for data in self.data)

    def next_batch(self, batch_size, assert_size=False):
        if self.current_pos == 0:
            self.shuffle_data()

        i_start = self.current_pos
        self.current_pos += batch_size
        if self.current_pos >= self.get_num_data():
            self.current_pos = 0

        result = (data[i_start:i_start+batch_size] for data in self.data)
        if assert_size:
            for data in result:
                assert data.shape[0] == batch_size
        yield result
