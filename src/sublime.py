from collections import namedtuple
import torch.nn as nn
from libs.sublime.main import Experiment
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.utils import to_torch_coo_tensor
from libs.sublime.utils import torch_sparse_to_dgl_graph
from dgl import graph
CONFIG = {
    #experimental
    'dataset': None, # to rewrite
    'ntrials': 5,
    'sparse': 1,
    'gsl_mode': 'structure_refinement', #['structure_inference', 'structure_refinement']
    'eval_freq': 5,
    'downstream_task': 'classification', #['classification', 'clustering'],
    'gpu': -1,
    'train_size': 0.7,
    'seed': 2104,
    #GCL Module
    'epochs': 1000,
    'lr': 0.01,
    'w_decay': .0,
    'hidden_dim': 512,
    'rep_dim': 64,
    'proj_dim': 64,
    'dropout': .5,
    'contrast_batch_size': 0,
    'nlayers': 2,
    # GCL - Augmentation
    'maskfeat_rate_learner': .2,
    'maskfeat_rate_anchor': .2,
    'dropedge_rate': .5,
    # GCL module
    'type_learner': 'fgp', #["fgp", "att", "mlp", "gnn"]
    'k': 30,
    'sim_function': 'cosine', #['cosine', 'minkowski']
    'gamma': .9,
    'activation_learner': 'relu', #["relu", "tanh"]
    # Evaluation Network (Classification),
    'epochs_cls': 200,
    'lr_cls': 1e-3,
    'w_decay_cls': 5e-4,
    'hidden_dim_cls': 32,
    'dropout_cls': .5,
    'dropedge_cls': .25,
    'nlayers_cls': 2,
    'patience_cls': 10,
    #structure bootstrapping
    'tau': 1,
    'c': 0,
}  
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu' 



class SublimeModule(nn.Module):
    def __init__(self,
                #   features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj,
                    config=None):
        """
        #forward pass parameters: features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj,
        """
        super().__init__()
        config = config if config is not None else CONFIG
        args = namedtuple('args', config.keys())

        self.args = args(**config)
        self.experiment = Experiment(self.args)

    @property
    def model(self):
        return self.experiment.model
    
    @property
    def graph_learner(self):
        return self.experiment.graph_learner
    
    def _gen_masks(self, n):
        train, test = train_test_split(torch.arange(0, n, 1),
                                        train_size=self.args.train_size, 
                                        random_state=self.args.seed)
        val, test = train_test_split(test, train_size=.5, random_state=self.args.seed)
        train_mask = torch.zeros((n,)).bool()
        val_mask = torch.zeros((n,)).bool()
        test_mask = torch.zeros((n,)).bool()
        train_mask[train] = True
        val_mask[val] = True
        test_mask[test] = True
        return train_mask, val_mask, test_mask
 
    def forward(self, features, edges, edges_weights=None, mask=None, label_col=-1):
        labels = features[:, label_col]
        nfeats = features.size(1)
        feat_mask = torch.ones((nfeats,)).bool()
        # feat_mask[label_col] = False
        features = features[:, feat_mask]
        # labels = (labels == 1).nonzero()[:, 1]
        nclasses = labels.unique().size(0)
        n = features.size(0)
        orig_adj  = torch_sparse_to_dgl_graph(to_torch_coo_tensor(edges, edge_attr=edges_weights))
        
        return self.experiment.train(
            self.args,
            features, nfeats, labels, nclasses, *self._gen_masks(n), orig_adj
        )



