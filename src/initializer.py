import torch
from torch_geometric.utils import (
    to_torch_coo_tensor, from_scipy_sparse_matrix, erdos_renyi_graph,
    barabasi_albert_graph, 
)
from itertools import product
from misc import discretize, wrong_arguments_display, NotFittedError, get_groups
from numpy import array, arange, unique
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
    

class BaseInitializer:
    __kwargs = frozenset()

    def __init__(self, *, device='cpu', **init_kwargs):
        if self._check_init_kwargs(init_kwargs):
            self.init_kwargs = init_kwargs
        self._fitted = False
        self.data = None
        self.init_kwargs = init_kwargs
        self.G = None
        self.device = device
        self._weights = None
        self._adjlist = None

    @wrong_arguments_display
    def _check_init_kwargs(self, init_kwargs):
        for val in init_kwargs:
            if val not in self.__kwargs:
                raise TypeError
        return True

    def __call__(self, data):
        self.data = data
        self._fitted = True
    
    def reset(self):
        self.__init__(self.device, **self.init_kwargs)

    def _doc(self):
        return self.__doc__

    @property
    def adjacency_matrix(self,):
        if not self._fitted:
            raise NotFittedError(type(self))
        return to_torch_coo_tensor(self._adjlist, self._weights).to(self.device)
    
    @property
    def edgelist(self):
        if not self._fitted:
            raise NotFittedError(type(self))
        return self._adjlist.to(self.device)
    
    @property
    def graph(self):
        if not self._fitted:
            raise NotFittedError(type(self))
        if self._weights is None:
            self._weights = torch.ones((len(self._adjlist),))

        return Data(
            x=torch.Tensor(self.data, device=self.device),
            edge_index=self._adjlist.to(self.device),
            edge_attr=self._weights.to(self.device)
        )


class KNNInit(BaseInitializer):
    """
    n_neighbors: int
    mode: ['connectivity', 'distance']
    metric: str = 'minkowski'| distance metric
    p: int | p-degree in minkowski metric
    """
    __kwargs = frozenset(['n_neighbors', 'mode', 'metric', 'p'])

    def __call__(self, data):
        super().__call__(data)
        spadj = kneighbors_graph(self.data, **self.init_kwargs)
        self._adjlist, self._weights = from_scipy_sparse_matrix(spadj)
        return self
      
        
class EpsilonRadiusInit(BaseInitializer):
    """
    radius: float - radius of hyper sphere
    """
    __kwargs = frozenset(['radius', 'mode', 'metric', 'p'])

    def __call__(self, data):
        super().__call__(data)
        spadj = radius_neighbors_graph(self.data, **self.init_kwargs)
        self._adjlist, self._weights = from_scipy_sparse_matrix(spadj)
        return self
        

class RandomInit(BaseInitializer):
    """
    random_graph: ["erdos_renyi", "barabasi_albert"] | Type of random graph 
    edge_prob: float in [0, 1] | Edge probability (approx. graph density)
    """
    def __call__(self, data):
        super().__call__(data)
        type_ = self.init_kwargs.pop('random_graph', 'erdos_renyi')
        if type_ == 'erdos_renyi':
            self._adjlist = erdos_renyi_graph(
                self.data.shape[0], **self.init_kwargs
            ).to(self.device)
        elif type_ == 'barabasi_albert':
            prob = self.init_kwargs.pop('edge_prob', 0)
            n = data.shape[0]
            self.init_kwargs['num_edges'] = int(n * prob)
            self._adjlist = barabasi_albert_graph(
                self.data.shape[0], **self.init_kwargs
            ).to(self.device)
        else:
            raise ValueError('random_graph should be in ["erdos_renyi", "barabasi_albert"]')
        return self
    
class FeatureCliqueInit(BaseInitializer):
    __kwargs = frozenset(['column_indices', 'division_factor', 'discretize_func', 'cols2disc'])

    def __call__(self, data):
        super().__call__(data)
        col_ind = self.init_kwargs['column_indices']
        df = self.init_kwargs['division_factor']
        discretize = self.init_kwargs['discretize_func']
        cols2disc = self.init_kwargs['cols2disc']
        edgelist = []
        for indices in get_groups(data, col_ind, discretize, cols2disc, df=df):
            edgelist.extend(x for x in product(indices, indices))
        self._adjlist = torch.tensor(edgelist).t().to(self.device)
        self._weights = torch.ones((len(edgelist),)).to(self.device)
        return self


if __name__ == '__main__':
    print('initializer.py')
    from pandas import read_csv
    data = read_csv(r'C:\Users\user\labevent\data\refined\wide\blood_chemistry_17.csv', 
                    index_col=(0, 1),
                    #   usecols=(7, 8, 9)
                    ).astype('float32')
    colinds = [1]
    print(data.columns)
    data = data.values

    G = FeatureCliqueInit(division_factor=5, column_indices=colinds, discretize_func=discretize)(data)
    print(G.edgelist)
    print(G.adjacency_matrix)
    print(G.graph)

