from missing_sampler import BaseMissingSampler
from initializer import BaseInitializer
import numpy as np
from pandas import DataFrame
from fp import FeaturePropagation
from tqdm import tqdm
from torch import tensor, float32, cuda
from sklearn.preprocessing import RobustScaler

DEVICE = 'cuda' if cuda.is_available() else 'cpu'

class Pipeline:    
    def __init__(self, preprocessor,
                 imputer,
                 criterion) -> None:
        self.preprocessor = preprocessor
        self.device =  DEVICE
        self.imputer = imputer.to(self.device)
        self._modules = {}
        self.loss_func = criterion.to(self.device)
        self._imputer = None
        self.edge_list = None
    

    def reset(self):
        for module in self._modules.values():
            if hasattr(module, 'reset'):
                module.reset()

    def _eval(self, data):
        dropped, struct, mask = self.preprocessor.run(data.detach().cpu(), 'edgelist')
        data = tensor(self.preprocessor._scaler.transform(
            data.cpu().numpy() if hasattr(data, 'cpu') else data
        )).to(self.device)
        mask = mask.to(self.device)
        imputed = self.imputer(dropped.to(self.device), 
                               struct.to(self.device),
                               mask.to(self.device)).to(self.device)
        loss = self.loss_func(imputed, data, mask)
        return loss


    def eval(self, data, n_trials,):
        losses = []
        data = tensor(data.to_numpy())
        for trial in tqdm(range(n_trials), leave=False, desc='Trial: '):  
            losses.append(self._eval(data))
        return losses
    
class Preprocessor():
    def __init__(self, sampler,
                 initializer,
                 scale=True):
        super().__init__()
        self._scaler = RobustScaler() if scale else None
        self._sampler = sampler #UniformMissing(droprate, target_columns, data_columns=data_columns)
        self._initializer = initializer

    def run(self, data, graph_repr='edgelist', dtype=float32, fill_value=0):
        if hasattr(data, 'cpu'):
            data = data.cpu()
        dropped = self._sampler.drop(data, fill_value)
        mask = self._sampler.to_tensor()
        dropped = tensor(self._scaler.fit_transform(dropped)\
                         if self._scaler is not None else dropped,
                         dtype=dtype)
        struct = getattr(self._initializer(dropped.cpu().numpy()), graph_repr)

        return dropped, struct, mask
