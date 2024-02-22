import pandas as pd
import numpy as np


class BaseMissingSampler:
    def __init__(self, data, droprate, target_cols=None, random_state=None, **read_kw):
        assert 0 <= droprate <= 1, 'Droprate is within [0, 1] range!'
        self.path = None
        self._mask = None
        self.rg = np.random.Generator(np.random.PCG64(random_state))
        self.len = 0
        self.done = False
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

    def _generate_mask(self):
        raise NotImplementedError

    def drop(self, inplace=False):
        self._generate_mask()
        self._data.loc[:, self.target_cols] = self._data[self.target_cols].where(self._mask, np.nan)
        return self

    def save(self, path):
        if path.endswith('.csv') and isinstance(self._data, pd.DataFrame):
            self._data.to_csv(path)
        else:
            raise NotImplementedError

    @property
    def droppedn(self):
        assert self.done, 'The dropout has\'nt been done yet!'
        return pd.Series(index=self.target_cols, data=self._mask.sum(0), name='Nan number')


class UniformMissing(BaseMissingSampler):
    def _generate_mask(self):
        self._mask = self.rg.random((self._data.shape[0], len(self.target_cols))) < self.droprate
        self.done = True
        return self._mask


if __name__ == '__main__':
    um = UniformMissing('../data/refined/wide/blood_chemistry_17.csv', .3,
                        target_cols=['Potassium',
                                     'Sodium', 'Creatinine', 'Chloride', 'Urea Nitrogen', 'Bicarbonate',
                                     'Anion Gap', 'Glucose', 'Magnesium', 'Calcium, Total', 'Phosphate']
                        )
    um.drop()
    print(um.droppedn)




