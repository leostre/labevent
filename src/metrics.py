from loss import Separable
import torch
from torchmetrics.regression import MeanAbsolutePercentageError
from torchmetrics.functional import mean_absolute_percentage_error

class SeparableMAPE(Separable, MeanAbsolutePercentageError):
    def _logic(self, input, target, mask):
        input, target = input * mask, target * mask
        errors = (target - input).abs_()
        errors /= errors.max(0).values
        errors[torch.isnan(errors)] = 0
        return errors.sum(0).div_(mask.sum(0))

    def forward(self, input, target, mask):
        out = super().forward(input, target, mask)
        out[torch.isnan(out)]  = 0
        return out
    
    
