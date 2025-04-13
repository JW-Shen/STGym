"""
Baseline method, STIDGCN [T-ITS, 2024].
Author: ChunWei Shen

Reference:
* Paper: https://ieeexplore.ieee.org/document/10440184
* Code: https://github.com/LiuAoyu1998/STIDGCN
"""
from typing import List, Dict, Any, Tuple, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from metadata import N_DAYS_IN_WEEK

from modeling.module.layers import STI
from modeling.module.common_layers import Linear2d, AuxInfoEmbeddings, GLU

class STIDGCN(nn.Module):
    """STIDGCN framework.

        Args:
            in_dim: input feature dimension
            in_len: input sequence length
            out_len: output sequence length
            n_series: number of series
            h_dim: hidden dimension
            kernel_size: kernel size
            dropout: dropout ratio
            tid_emb_dim: dimension of time in day embedding
            diw_emb_dim: dimension of day in week embedding
            n_tids: number of times in day
    """
    
    def __init__(
        self,
        in_dim: int,
        in_len: int,
        out_len: int,
        n_series: int,
        st_params: Dict[str, Any],
    ) -> None:
        self.name = self.__class__.__name__
        super(STIDGCN, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        h_dim = st_params["h_dim"]
        kernel_size = st_params["kernel_size"]
        n_adjs = st_params["n_adjs"]
        gcn_depth = st_params["gcn_depth"]
        dropout = st_params["dropout"]
        self.n_tids = st_params["n_tids"]
        self.tid_emb_dim = h_dim
        self.diw_emb_dim = h_dim
        self.in_len = in_len
        self.in_dim = in_dim
        self.n_series = n_series

        # Model blocks
        # Encoder
        # Input linear layer
        self.in_lin = Linear2d(in_dim, h_dim)
        # Auxiliary information embeddings
        self.aux_info_emb = AuxInfoEmbeddings(
            n_tids=self.n_tids,
            tid_emb_dim=self.tid_emb_dim,
            diw_emb_dim=self.diw_emb_dim
        )
        # STI Tree
        self.tree = STITree(
            h_dim=h_dim * 2,
            kernel_size=kernel_size,
            n_adjs=n_adjs,
            gcn_depth=gcn_depth,
            n_series=n_series,
            dropout=dropout
        )

        # Decoder
        self.glu = GLU(h_dim=h_dim * 2, dropout=dropout)
        # Output layer
        self.output = nn.Conv2d(in_channels=h_dim * 2, out_channels=out_len, kernel_size=(1, out_len))

    def forward(
        self, x: Tensor, As: Optional[List[Tensor]] = None, **kwargs: Any
    ) -> Tuple[Tensor, Union[Tensor, None], None]:
        """Forward pass.

        Args:
            x: input sequence

        Shape:
            x: (B, P, N, C)
        """
        if self.tid_emb_dim > 0:
            tid = (x[..., 1] * self.n_tids).long()
        if self.diw_emb_dim > 0:
            diw = (x[..., 2] * N_DAYS_IN_WEEK).long()
        x = x.transpose(1, 3)

        # Encoder
        # Auxiliary information embeddings
        _, _, x_tid, x_diw, _ = self.aux_info_emb(tid=tid, diw=diw)
        x_time = (x_tid + x_diw).permute(0, 3, 2, 1)      # (B, D, N, T)
        h = torch.cat([self.in_lin(x), x_time], dim=1)
        # STI Tree
        h = self.tree(h)

        # Decoder
        h = self.glu(h) + h
        output = self.output(F.relu(h)).squeeze(-1)

        return output, None, None
    

class STITree(nn.Module):
    """Spatial-Temporal Interaction Tree.

        Args:
            h_dim: hidden dimension
            kernel_size: kernel size
            gcn_depth: depth of graph convolution
            n_series: number of nodes
            dropout: dropout ratio
            split: if True, apply split
    """
    
    def __init__(
        self,
        h_dim: int,
        kernel_size: int,
        n_adjs: int,
        gcn_depth: int,
        n_series: int,
        dropout: float,
    ) -> None:
        super(STITree, self).__init__()

        # Model blocks
        self.memory1 = nn.Parameter(torch.randn(h_dim, n_series, 6))
        self.memory2 = nn.Parameter(torch.randn(h_dim, n_series, 3))
        self.memory3 = nn.Parameter(torch.randn(h_dim, n_series, 3))

        self.STI1 = STI(
            h_dim=h_dim,
            kernel_size=kernel_size,
            n_adjs=n_adjs,
            gcn_depth=gcn_depth,
            n_series=n_series,
            emb=self.memory1,
            dropout=dropout
        )

        self.STI2 = STI(
            h_dim=h_dim,
            kernel_size=kernel_size,
            n_adjs=n_adjs,
            gcn_depth=gcn_depth,
            n_series=n_series,
            emb=self.memory2,
            dropout=dropout
        )

        self.STI3 = STI(
            h_dim=h_dim,
            kernel_size=kernel_size,
            n_adjs=n_adjs,
            gcn_depth=gcn_depth,
            n_series=n_series,
            emb=self.memory3,
            dropout=dropout
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: input sequence
        """
        h_even1, h_odd1 = self.STI1(x)
        h_even2, h_odd2 = self.STI2(h_even1)
        h_even3, h_odd3 = self.STI3(h_odd1)

        h1 = self.concat_and_realign(h_even2, h_odd2)
        h2 = self.concat_and_realign(h_even3, h_odd3)
        h = self.concat_and_realign(h1, h2)

        h = h + x

        return h
    
    def concat_and_realign(self, x_even: Tensor, x_odd: Tensor) -> Tensor:
        """Concat & Realign.

        Args:
            x_even: even sub-sequence
            x_odd: odd sub-sequence

        Shape:
            x_even: (B, C, N, L)
            x_odd: (B, C, N, L)
            output: (B, C, N, L')
        """

        x_even = x_even.permute(3, 1, 2, 0)    # (L, C, N, B)
        x_odd = x_odd.permute(3, 1, 2, 0)      # (L, C, N, B)

        even_len = x_even.shape[0]
        odd_len = x_odd.shape[0]
        min_len = min((odd_len, even_len))

        output = []
        for i in range(min_len):
            output.append(x_even[i].unsqueeze(0))
            output.append(x_odd[i].unsqueeze(0))
        
        if odd_len < even_len: 
            output.append(x_even[-1].unsqueeze(0))

        output = torch.cat(output, dim=0).permute(3, 1, 2, 0)

        return  output