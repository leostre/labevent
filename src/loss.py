from torch import nn
import torch
class SeparableMSE(nn.MSELoss):
    def __init__(self, sep=True, target_columns=None, agg=None):
        super().__init__()
        self.sep = sep
        self.target_columns = target_columns
        self.agg = agg
    
    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor, mask: torch.BoolTensor):
        assert (input.size() == target.size()), 'Can compute only tensors with identical shapes!'
        assert input.size() == mask.size(), 'Mask size need to match mask size!'

        if self.sep:
            errors = ( #getattr(torch, self.agg)
                torch.mul(torch.square(input - target), mask).sum(dim=0).div(mask.sum(0))
            )
            errors[torch.isnan(errors)] = 0
            return errors
        else:
            return super().forward(input[mask], target[mask])


if __name__ == '__main__':
    x = torch.FloatTensor([
            [1., 2.],
            [4., 6.,],
            [9., -1.]
        ])
    y = torch.FloatTensor(
            [
                [1., 0.],
            [4., 3.,],
            [9., -2.]
            ]
        )
    
    x = torch.add(x, 4)
    y = torch.mul(y, 3)
    loss = (SeparableMSE(False)(
        x, y
    ))
    loss.backward()
    print(loss.item())

        
