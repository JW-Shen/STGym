"""
Baseline method, NLinear [AAAI, 2023].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2205.13504
* Code: https://github.com/cure-lab/LTSF-Linear
"""
from typing import List, Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

class NLinear(nn.Module):
    """NLinear framework.

    Args:
        in_len: input sequence length
        out_len: output sequence length
        n_series: number of series
        individual: whether to share linear layer for all nodes
    """
    def __init__(self, in_len: int, out_len: int, st_params: Dict[str, Any]) -> None:
        self.name = self.__class__.__name__
        super(NLinear, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.out_len = out_len
        # Spatio-temporal pattern extractor
        self.n_series = st_params["n_series"]
        self.individual = st_params["individual"]
        
        # Model blocks
        # Linear layer
        if self.individual:
            self.Linear = nn.ModuleList()
            for _ in range(self.n_series):
                self.Linear.append(nn.Linear(in_len, self.out_len))
        else:
            self.Linear = nn.Linear(in_len, self.out_len)

    def forward(self, x: Tensor, As: Optional[List[Tensor]] = None, **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Args:
            x: input sequence

        Shape:
            x: (B, P, N, C)
            output: (B, Q, N)
        """
        x = x[..., 0]   # (B, P, N)
        # Normalization
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        # Linear layer
        if self.individual:
            output = torch.zeros([x.size(0), self.out_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.n_series):
                output[:, :, i] = self.Linear[i](x[:, :, i])
        else:
            output = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        output = output + seq_last

        return output, None, None