from typing import Any


class BaseImputer:
    def impute(self, inplace=True):
        raise NotImplemented
    

class FPImp(BaseImputer):
    def __init__(self, n_iterations):
        self.n_iterations = n_iterations

    