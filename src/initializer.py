import torch
from torch_geometric.utils import (
    to_torch_coo_tensor, from_scipy_sparse_matrix, erdos_renyi_graph,
    barabasi_albert_graph
)
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
# from utils import wrong_arguments_display

# def tab2graph(path, init_method, **init_method_kwargs):
#     tabular = pd.read_csv(path)
#     adj = init_method
    
def wrong_arguments_display(func):
    def wrap(self, *args, **kwargs):
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


class BaseInitializer:
    def __init__(self, data, device='cpu', **init_kwargs):
        self.data = data
        self.init_kwargs = init_kwargs
        self.G = None
        self.device = device
        self.weights = None
        self.adjlist = None
        self._gen()

    @wrong_arguments_display
    def _gen(self):
        raise NotImplementedError('It\'s a base class!')

    def _doc(self):
        return self.__doc__

    @property
    def adjacency_matrix(self,):
        return to_torch_coo_tensor(self.adjlist, self.weights).to(self.device)
    
    @property
    def edgelist(self):
        return self.adjlist.to(self.device)
    
    @property
    def graph(self):
        if not self.weights:
            self.weights = torch.ones((len(self.adjlist),))
        return Data(
            x=torch.Tensor(self.data, device=self.device),
            edge_index=self.adjlist.to(self.device),
            edge_attr=self.weights.to(self.device)
        )


class SelfLoop(BaseInitializer):
    def _gen(self):
        pass


class KNNInit(BaseInitializer):
    """
    n_neighbors: int
    mode: ['connectivity', 'distance']
    """
    @wrong_arguments_display
    def _gen(self):
        spadj = kneighbors_graph(self.data, **self.init_kwargs)
        self.adjlist, self.weights = from_scipy_sparse_matrix(spadj)
      
        
class EpsilonRadiusInit(BaseInitializer):
    """
    eps: float - radius of hyper sphere
    """
    @wrong_arguments_display
    def _gen(self):
        spadj = radius_neighbors_graph(self.data, **self.init_kwargs)
        self.adjlist, self.weights = from_scipy_sparse_matrix(spadj)
        

class RandomInit(BaseInitializer):
    """
    random_graph: ["erdos_renyi", "barabasi_albert"] | Type of random graph 
    edge_prob: float in [0, 1] | Edge probability (approx. graph density)
    """
    @wrong_arguments_display
    def _gen(self):
        type_ = self.init_kwargs.pop('random_graph', 'erdos_renyi')
        if type_ == 'erdos_renyi':
            self.adjlist = erdos_renyi_graph(
                self.data.shape[0], **self.init_kwargs
            ).to(self.device)
        elif type_ == 'barabasi_albert':
            prob = self.init_kwargs.pop('edge_prob', 0)
            n = self.data.shape[0]
            self.init_kwargs['num_edges'] = int(n * prob)
            self.adjlist = barabasi_albert_graph(
                self.data.shape[0], **self.init_kwargs
            ).to(self.device)
        else:
            raise ValueError('random_graph should be in ["erdos_renyi", "barabasi_albert"]')


if __name__ == '__main__':
    print('initializer.py')
    from pandas import read_csv
    data = read_csv(r'C:\Users\user\labevent\data\refined\wide\blood_chemistry_17.csv', 
                    index_col=(0, 1), usecols=(7, 8, 9)).astype('float32').values
    G = RandomInit(data, random_graph='erdos_renyi', edge_prob=.5)
    print(G.edgelist)
    print(G.adjacency_matrix)
    print(G.graph)

