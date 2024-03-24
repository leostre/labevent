from missing_sampler import BaseMissingSampler
from initializer import BaseInitializer
import numpy as np
from pandas import DataFrame
from fp import FeaturePropagation
from tqdm import tqdm
from torch import tensor 

class Pipeline:    
    def __init__(self) -> None:
        self._modules = {}
        self._initializer: BaseInitializer = None
        self._sampler: BaseMissingSampler = None
        self._loss_func = None
        self._imputer = None
        self.edge_list = None
        self._raw_data = None
        pass
      
    def initialize(self, initializer):
        self._initializer = initializer

    def reset(self):
        for module in self._modules.values():
            if hasattr(module, 'reset'):
                module.reset()

    def set_sampler(self, sampler):
        self._sampler = sampler
    
    def set_loss_func(self, loss_func):
        self._loss_func = loss_func
    
    def impute(self, imputer):
        self._imputer = imputer

    def eval(self, data: DataFrame, n_trials, n_iterations=8):
        losses = []
        data = tensor(data.to_numpy())
        for trial in tqdm(range(n_trials)):
            mask = self._sampler.to_tensor(new=True)
            dropped = self._sampler.drop(inplace=False, fill_value=0)
            dropped = tensor(dropped.values).float()
            self._initializer = self._initializer(dropped)
            fp = FeaturePropagation(num_iterations=n_iterations)
            imputed = fp.propagate(dropped, edge_index=self._initializer.edgelist, mask=mask)
            loss = self._loss_func(imputed[mask], data[mask]).item()
            losses.append(loss)
        return losses