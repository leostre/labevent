from missing_sampler import BaseMissingSampler
from initializer import BaseInitializer
import numpy as np
from pandas import DataFrame
from fp import FeaturePropagation
from tqdm import tqdm
from torch import tensor, float32
from sklearn.preprocessing import RobustScaler

class Pipeline:    
    def __init__(self, preprocessor,
                 imputer,
                 criterion, to_device='cpu') -> None:
        self.preprocessor = preprocessor
        self.imputer = imputer
        self._modules = {}
        self.loss_func = criterion
        self._imputer = None
        self.edge_list = None
        self.device = to_device
    

    def reset(self):
        for module in self._modules.values():
            if hasattr(module, 'reset'):
                module.reset()

    def _eval(self, data: DataFrame):
        dropped, struct, mask = self.preprocessor.run(data, 'edgelist')
        data = tensor(self.preprocessor._scaler.transform(data), device=self.device)
        imputed = self.imputer(dropped, struct, mask)
        loss = self.loss_func(imputed, data, mask)
        return loss


    def eval(self, data, n_trials,):
        losses = []
        for trial in tqdm(range(n_trials), leave=False, desc='Trial: '):
            data = tensor(data.to_numpy()).to(self.device)
            losses.append(self._eval(data).item())
        return losses
    
class Preprocessor():
    def __init__(self, sampler,
                 initializer,
                 scale=True, device='cpu'):
        super().__init__()
        self._scaler = RobustScaler() if scale else None
        self._sampler = sampler #UniformMissing(droprate, target_columns, data_columns=data_columns)
        self._initializer = initializer
        self.device = device

    def run(self, data, graph_repr='edgelist', dtype=float32, fill_value=0):            
        dropped = self._sampler.drop(data, fill_value)
        mask = self._sampler.to_tensor().to(self.device)
        dropped = tensor(self._scaler.fit_transform(dropped) if self._scaler is not None else dropped, dtype=dtype).to(self.device)
        struct = getattr(self._initializer(dropped), graph_repr).to(self.device)

        return dropped, struct, mask