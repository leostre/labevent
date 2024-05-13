import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv, GCNConv
    
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm.notebook import tqdm, trange

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from torch.nn.parameter import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PassThroughImputer(nn.Module):
    def forward(self, x, edges, mask):
        return x

def ex_relu(mu, sigma):
    is_zero = (sigma == 0)
    sigma[is_zero] = 1e-10
    sqrt_sigma = torch.sqrt(sigma)
    w = torch.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (torch.div(torch.exp(torch.div(- w * w, 2)), np.sqrt(2 * np.pi)) +
                              torch.div(w, 2) * (1 + torch.erf(torch.div(w, np.sqrt(2)))))
    nr_values = torch.where(is_zero, F.relu(mu), nr_values)
    return nr_values


def init_gmm(features, n_components):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    init_x = imp.fit_transform(features)
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag').fit(init_x)
    return gmm

class GCNmfConv(nn.Module):
    def __init__(self, in_features, out_features, n_components, dropout, bias=True):
        super(GCNmfConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.dropout = dropout
        self.conv = SimpleConv(aggr='add', combine_root='self_loop')
        self.logp = Parameter(torch.FloatTensor(n_components))
        self.means = Parameter(torch.FloatTensor(n_components, in_features))
        self.logvars = Parameter(torch.FloatTensor(n_components, in_features))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.gmm = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
    
    def reset_parameters(self, X):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        self.gmm = init_gmm(X, self.n_components) # impute mean values with simple imputer
        self.logp.data = torch.FloatTensor(np.log(self.gmm.weights_)).to(device)
        self.means.data = torch.FloatTensor(self.gmm.means_).to(device)
        self.logvars.data = torch.FloatTensor(np.log(self.gmm.covariances_)).to(device)

    def calc_responsibility(self, mean_mat, variances):
        dim = self.in_features
        log_n = (- 1 / 2) *\
            torch.sum(torch.pow(mean_mat - self.means.unsqueeze(1), 2) / variances.unsqueeze(1), 2)\
            - (dim / 2) * np.log(2 * np.pi) - (1 / 2) * torch.sum(self.logvars)
        log_prob = self.logp.unsqueeze(1) + log_n
        return torch.softmax(log_prob, dim=0)

    def forward(self, x, edges, mask, edge_weight=None):
        if self.gmm is None:
            self.reset_parameters(x)
        assert x.size() == mask.size()
        x_imp = x.repeat(self.n_components, 1, 1)
        x_isnan = mask.repeat(self.n_components, 1, 1)
        # x_isnan = torch.isnan(x_imp)
        variances = torch.exp(self.logvars)
        mean_mat = torch.where(x_isnan, self.means.repeat(x.size(0), 1, 1).permute(1, 0, 2), x_imp)
        var_mat = torch.where(x_isnan,
                              variances.repeat(x.size(0), 1, 1).permute(1, 0, 2),
                              torch.zeros(x_isnan.size(), device=device, requires_grad=True))
        # dropout
        dropmat = F.dropout(torch.ones_like(mean_mat), self.dropout, training=self.training)
        mean_mat = mean_mat * dropmat
        var_mat = var_mat * dropmat

        transform_x = torch.matmul(mean_mat, self.weight)
        if self.bias is not None:
            transform_x = torch.add(transform_x, self.bias)
        transform_covs = torch.matmul(var_mat, self.weight * self.weight)
        conv_x = []
        conv_covs = []

        for component_x in transform_x:
            conv_x.append(
                self.conv(component_x, edges, edge_weight=edge_weight)
                )
        for component_covs in transform_covs:
            conv_covs.append(
                self.conv(component_covs, edges, edge_weight=edge_weight)
                )
        transform_x = torch.stack(conv_x, dim=0)
        transform_covs = torch.stack(conv_covs, dim=0)
        expected_x = ex_relu(transform_x, transform_covs)

        # calculate responsibility
        gamma = self.calc_responsibility(mean_mat, variances)
        expected_x = torch.sum(expected_x * gamma.unsqueeze(2), dim=0)
        return expected_x

class Decoder(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        assert len(dims) >= 2, 'Specify at least input and output dimensions!'
        self.layers = nn.ModuleList(
            nn.Linear(dims[i - 1], dims[i]) for i in range(1, len(dims))
        )
        self.n = len(dims)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers, 1):
            x = layer(x)
            if i != self.n:
                x = F.leaky_relu(x, 0.2)

        return x
    
class Encoder(nn.Module):
    def __init__(self, *dims, linn=0, n_comp=3, dropout=.1):
        assert len(dims) >=3
        super().__init__()
        self.n = len(dims)
        self.linn = linn
        layers = [GCNmfConv(dims[0], dims[1], n_comp, dropout)]
        for i in range(2, len(dims)):
            layers.append(GCNConv(dims[i - 1], dims[i], add_self_loops=True))
        self.layers = nn.ModuleList(layers)

    
    def forward(self, x, edges, mask):
        for i, layer in enumerate(self.layers):
            if not i:
                x = F.leaky_relu(layer(x, edges, mask), .2)
            else:
                x = F.leaky_relu(layer(x, edges), .2)
        return x 

class GAEMF(nn.Module):
    def __init__(self, enc_sizes, dec_sizes, linn=1, dropout=.2, n_comp=4,):
        super().__init__()
        self.enc = Encoder(*enc_sizes, dropout=dropout, n_comp=n_comp, linn=linn)
        self.dec = Decoder(*dec_sizes)

    def forward(self, x, edges, mask):
        x = self.enc(x, edges, mask)
        x = self.dec(x)
        return x

def train(model, dataloaders, criterion, opt, sch, num_epochs, verbose=True):
    losses = {
        'train': [],
        'val': [],
    }
    for epoch in trange(num_epochs, desc='Epoch: '):
        for phase in ['train', 
                      'val'
                      ]:
            curloss = []
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for subg in tqdm(dataloaders[phase], desc='Nodes: '):
                x, edges = subg.x, subg.edge_index
                xmask = ~torch.isnan(x)
                if phase == 'train':
                    opt.zero_grad()
                    output = model(x, edges)
                    loss = criterion(output[xmask], x[xmask])
                else:
                    with torch.no_grad():
                        output = model(x, edges)
                        loss = criterion(output[xmask], x[xmask])
                if phase == 'train':
                    loss.backward()
                    opt.step()
                    sch.step()
                curloss.append(loss.item())

            losses[phase].append(sum(curloss) / len(curloss))
        if verbose:
            print(f'Epoch: {epoch} Train loss: {losses["train"][-1]} Val loss: {losses["val"][-1]}')
    return losses

class GCNmfImputer(nn.Module):
    def __init__(self, model, train_epochs=2, lr=0.01):
        super().__init__()
        self.model = model
        self.train_epochs = train_epochs
        self.lr = lr

    def __train(self, dataloader, criterion, opt, sch, num_epochs, verbose=True):
        losses = []
        self.model.train()
        for epoch in range(num_epochs):
            curloss = []
            for subg in dataloader:
                x, edges = subg.x, subg.edge_index
                xmask = ~torch.isnan(x)
                opt.zero_grad()
                output = self.model(x, edges, xmask)
                loss = criterion(output[xmask], x[xmask])
                loss.backward()
                opt.step()
                if sch:
                    sch.step()
                curloss.append(loss.item())
            losses.append(sum(curloss) / len(curloss))
        return losses
    
    def forward(self, x, edges, mask, pretrained=False, nneighbors=None):
        # mask = torch.isnan(x) if mask is None else mask
        if not pretrained:
            nneighbors = nneighbors if nneighbors is not None else [6, 6]
            criterion = nn.MSELoss()
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            dataloader = NeighborLoader(Data(x, edges), num_neighbors=nneighbors, batch_size=24)
            self.__train( 
                  dataloader,
                  criterion,
                  opt,
                  None,
                  num_epochs=self.train_epochs
                  )
            del criterion
            del opt
            del dataloader
            
        ret = self.model(x, edges, mask)
        ret[~mask] = x[~mask]
        return ret


"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import torch
from torch import Tensor
# from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add

def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD


class FeaturePropagation(torch.nn.Module):
    def __init__(self, n_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = n_iterations

    def forward(self, x: Tensor, edge_index, mask: Tensor) -> Tensor:
        # out is inizialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        adj = self.get_propagation_matrix(out, edge_index, n_nodes)
        for _ in range(self.num_iterations):
            # Diffuse current features
            out = torch.sparse.mm(adj, out)
            # Reset original known features
            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, x, edge_index, n_nodes):
        # Initialize all edge weights to ones if the graph is unweighted)
        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

        return adj
