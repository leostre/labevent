import numpy as np
from pandas import DataFrame, qcut

def wrong_arguments_display(func):
    def wrap(self, *args, **kwargs):
        res = None
        try:
            res = func(self, *args, **kwargs)
            return res
        except TypeError as err:
            err.args = err.args[0] + f'''
            Wrong arguments were passed to {func.__name__} of {str(type(self)).split('.')[-1][:-2]}
            Read the docs: 
            {self._doc()}''',
            raise 
        finally:
            return res
    return wrap

def discretize_by_div(x: np.array, df):
    return (x // df) * df

def discretize(x, df):
    return qcut(x, df)

def get_groups(data, col_inds, discretize_func=None, cols2disc=None, **disc_kwargs):
    if not discretize_func:
        discretize_func = discretize
    cols = list(range(len(col_inds)))
    data = DataFrame(data,)
    if not cols2disc is None:
        data.iloc[:, cols2disc] = data.iloc[:, cols2disc].apply(discretize_func, **disc_kwargs)
    data = data.iloc[:, col_inds]
    data.columns = cols
    data['index'] = data.index
    groups = data.groupby(cols)
    for _, gr in groups:
        yield gr['index'].values

    
    
class NotFittedError(Exception):
    def __init__(self, type_, *args: object) -> None:
        super().__init__(*args)
        self.args = (f'Object of {type_} has not been fitted',)