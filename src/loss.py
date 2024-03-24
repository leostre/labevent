from torch import nn
import torch
class SeparableMSE(nn.MSELoss):
    def __init__(self, sep=True, target_columns=None):
        super().__init__()
        self.sep = sep
        self.target_columns = target_columns
    
    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor):
        assert (input.size() == target.size()), 'Can compute only tensors with identical shapes!'
        if self.sep:
            errors = torch.square(input - target).mean(dim=0)
            return errors
        else:
            return super().forward(input, target)

        

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

        
