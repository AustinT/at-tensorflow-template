import numpy as np


class BaseDataGen:
    def __init__(self, config):
        self.config = config
        self.data = self.load_data()
        self.current_pos = {key: 0 for key in self.data}

    def load_data(self):
        raise NotImplementedError

    def shuffle_data(self, key='train'):
        raise NotImplementedError

    def get_num_data(self, key='train'):
        raise NotImplementedError

    def random_batch(self, batch_size, key='train'):
        indices = np.random.choice(self.get_num_data(key=key), batch_size)
        return (data[indices] for data in self.data[key])

    def next_batch(self, batch_size, key='train', assert_size=False):
        if self.current_pos[key] == 0:
            self.shuffle_data(key=key)

        i_start = self.current_pos[key]
        self.current_pos[key] += batch_size
        if self.current_pos[key] >= self.get_num_data(key):
            self.current_pos[key] = 0

        result = (data[i_start:i_start+batch_size] for data in self.data[key])
        if assert_size:
            for data in result:
                assert data.shape[0] == batch_size
        return result
