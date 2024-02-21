import pandas as pd
import numpy as np
class BaseMissingSampler:
    def __init__(self):
        self.path = None
        self._data = None
        self.mask = None
        # self.random_state = random_state
        self.target_cols = []
        self.len = 0
        self.done = False

    def __len__(self):
        return self.len

    def scan_data(self, data, target_cols=None):
        if target_cols:
            self.target_cols = target_cols
        if isinstance(data, str):
            if data.endswith('.csv'):
                self._data = pd.read_csv(data)
            else:
                raise NotImplementedError
        else:
            self._data = data

        if hasattr(self._data, 'shape'):
            self.len = self._data.shape[0]

    def drop(self, droprate, inplace=False):
        assert 0 <= droprate <= 1, 'Droprate is within [0, 1] range!'
        self.done = True

    def save(self, path):
        if path.endswith('.csv') and isinstance(self._data, pd.DataFrame):
            self._data.to_csv(path)
        else:
            raise NotImplementedError

    @property
    def droppedn(self):
        assert self.done, 'The dropout has\'nt been done yet!'
        length = len(self.mask)
        if length > 1:
            return {col: self.mask[col].sum() for col in self.target_cols}
        elif length:
            return self.mask.values()[0]
        else:
            return 0

    @property
    def masks(self):
        return self.mask


class UniformMissing(BaseMissingSampler):
    def drop(self, droprate, inplace=False):
        super().drop(droprate)
        for col in self.target_cols:
            self.mask[col] = (np.random.uniform(0, 1, size=(self.len,)) < droprate)
        if inplace:
            self.loc[self.mask[col], self.target_cols] = np.nan
        else:
            res = self._data.copy()
            res.loc[self.mask[col], self.target_cols] = np.nan
            return res




