import numpy as np
import torch
import torch.nn as nn

class Policy_Controller_base(nn.Module):
    def __init__(self, *args) -> None:
        super(Policy_Controller_base, self).__init__()
    def forward(self, obs_s:list[np.ndarray], is_batch = False):
        raise NotImplementedError("error")
    @property
    def K(self)->torch.Tensor:
        raise NotImplementedError("K not implemented")