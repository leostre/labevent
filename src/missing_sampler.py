import pandas as pd
import numpy as np
from torch import tensor
from misc import columns_str2int

class BaseMissingSampler:
    def __init__(self, droprate, shape=None, target_cols=None, random_state=None, data_columns=None):
        assert 0 <= droprate <= 1, 'Droprate is within [0, 1] range!'
        self.path = None
        self._mask = None
        self.rg = np.random.Generator(np.random.PCG64(random_state))
        self._shape = shape
        self._done = False
        self.droprate = droprate
        self.data_columns = data_columns
        self.target_cols = target_cols
        self._target_cols_ind = columns_str2int(self.data_columns, target_cols)

    @property
    def mask(self):
        assert self._done, 'The dropout has\'nt been done yet!'
        return self._mask
    

    def drop(self, data, fill_value=np.nan, *, add_indicator=False, inplace=False):
        self._shape = data.shape # extend for .size
        self._done = True
        self.generate_mask(self._shape)
        X = data.copy() if not inplace else data
        X[self._mask] = fill_value

        if add_indicator:
            assert not inplace, 'Sorry, inplace addition of columns is not supported!'
            if fill_value == np.nan:
                hmm =  self._mask.isnan().sum(1)
            else:
                hmm = (self._mask == fill_value).sum(1)
            X = np.hstack([X, hmm.reshape(-1, 1)])
            self.data_columns.append('how_many_missing')
        if not inplace:
            return X
        return self
    
    @property
    def done(self):
        return self._done
    
    def _gen(self):
        assert self._shape is not None and self.done, 'Specify the mask shape!'
        self._mask = np.zeros(shape=self._shape).astype(bool)

    @property
    def droppedn(self):
        assert self._done, 'The dropout has\'nt been done yet!'
        return dict(*zip(self.target_cols, self._mask.sum(0)))
    
    def to_tensor(self, *, new=True):
        if new or not self._done:
            self._gen()
        assert self._done, 'The dropout has\'nt been done yet!'
        return tensor(self._mask, dtype=bool)

    def generate_mask(self, shape=None, new=True):
        if shape:
            self._shape = shape
        if new or not self._done:
            self._gen()
        return self._mask


class UniformMissing(BaseMissingSampler):
    def _gen(self):
        super()._gen()
        self._mask[:, self._target_cols_ind] = self.rg.random((self._shape[0], 
                                                            len(self._target_cols_ind))) < self.droprate
        self._done = True


if __name__ == '__main__':
    import os
    # os.chdir('src')
    print('missing_sampler.py')
    # um = UniformMissing('../data/refined/wide/blood_chemistry_17.csv', .3,
    #                     target_cols=['Potassium',
    #                                  'Sodium', 'Creatinine', 'Chloride', 'Urea Nitrogen', 'Bicarbonate',
    #                                  'Anion Gap', 'Glucose', 'Magnesium', 'Calcium, Total', 'Phosphate']
    #                     )
    # um.generate_mask()
    # print(um.to_tensor())




