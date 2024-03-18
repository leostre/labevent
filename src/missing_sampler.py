import pandas as pd
import numpy as np
from torch import tensor

class BaseMissingSampler:
    def __init__(self, data, droprate, target_cols=None, random_state=None, **read_kw):
        assert 0 <= droprate <= 1, 'Droprate is within [0, 1] range!'
        self.path = None
        self._mask = None
        self.rg = np.random.Generator(np.random.PCG64(random_state))
        self.len = 0
        self._done = False
        self.droprate = droprate
        self._data: pd.DataFrame = None
        self._scan_data(data, **read_kw)
        if target_cols:
            self.target_cols = target_cols
        else:
            self.target_cols = self._data.columns

    def __len__(self):
        return self.len

    @property
    def mask(self):
        return self._mask

    def _scan_data(self, data, **read_kw):
        if isinstance(data, str):
            if data.endswith('.csv'):
                self._data = pd.read_csv(data, **read_kw)
            elif data.endswith('.parquet'):
                self._data = pd.read_parquet(data, **read_kw)
            else:
                raise NotImplementedError('Only .csv and .parquet are supported')
            self.path = data
        elif isinstance(data, pd.DataFrame):
            self._data = data
        else:
            raise ValueError('Only pandas.DataFrame objects and 2D-tables in .csv/.parquet are supported')
        self.len = self._data.shape[0]
    
    # def gen_mask(self):
    #     self._generate_mask()

    def drop(self, fill_value=np.nan, *, add_indicator=False, inplace=False):
        if not self._done:
            self.generate_mask()
        X = self._data.copy() if not inplace else self._data
        X.loc[:, self.target_cols] = X[self.target_cols].where(~self._mask, fill_value)
        if add_indicator:
            if fill_value == np.nan:
                X.loc[:, 'how_many_missing'] = self._mask.isna().sum(1)
            else:
                X.loc[:, 'how_many_missing'] = (self._mask == fill_value).sum(1)
        if not inplace:
            return X
        return self

    def save(self, path):
        if path.endswith('.csv') and isinstance(self._data, pd.DataFrame):
            self._data.to_csv(path)
        else:
            raise NotImplementedError
        return self
    
    @property
    def done(self):
        return self._done

    @property
    def droppedn(self):
        assert self._done, 'The dropout has\'nt been done yet!'
        return pd.Series(index=self.target_cols, data=self._mask.sum(0), name='Nan number')
    
    def to_tensor(self, *, full=True, new=True):
        if new or not self._done:
            self._gen()
        assert self._done, 'The dropout has\'nt been done yet!'
        if not full:
            return tensor(self._mask, dtype=bool)
        full_mask = np.zeros((self._data.shape))
        for i, col in enumerate(self._data.columns):
            if col in self.target_cols:
                full_mask[:, i] = self._mask[:, self.target_cols.index(col)]
        return tensor(full_mask.astype(bool))

    def generate_mask(self, new=True):
        if new or not self._done:
            self._gen()
        return self._mask


class UniformMissing(BaseMissingSampler):
    def _gen(self):
        self._mask = self.rg.random((self._data.shape[0], len(self.target_cols))) < self.droprate
        self._done = True




if __name__ == '__main__':
    import os
    # os.chdir('src')
    print('missing_sampler.py')
    um = UniformMissing('../data/refined/wide/blood_chemistry_17.csv', .3,
                        target_cols=['Potassium',
                                     'Sodium', 'Creatinine', 'Chloride', 'Urea Nitrogen', 'Bicarbonate',
                                     'Anion Gap', 'Glucose', 'Magnesium', 'Calcium, Total', 'Phosphate']
                        )
    um.generate_mask()
    print(um.to_tensor())




