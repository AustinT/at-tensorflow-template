import numpy as np
from base.base_data_gen import BaseDataGen


class X2Data(BaseDataGen):

    def load_data(self):
        x = np.random.random((self.get_num_data(), 1))*2 - 1
        y = x**2

        return {"train": (x, y)}

    def get_num_data(self, key=None):
        return 2000

    def shuffle_data(self, key=None):
        self.data = self.load_data()
