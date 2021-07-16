from typing import Optional, Tuple

import torch
from torch_sparse.tensor import SparseTensor


def weighted_sample(weights: torch.Tensor, rowptr: torch.Tensor, rowcount: torch.Tensor, num_neighbors: int) -> torch.Tensor:
    """
    samples index based on non negative weights provided in the adj tensor.
    This is much more faster than torch.Multinomial because it works with SparseTensor
    :param weights: tensor of weights
    :param rowptr: row index of nodes for which we need to find neighbors
    :param rowcount: number of neighbors for each node in rowptr
    :return: tensor of node index of sampled neighbors
    """
    start_index = rowptr
    end_index = rowptr + rowcount
    result = []
    for i,j in zip(start_index, end_index):
        wt_sample = weights[i:j]
        wt_sample = torch.cumsum(wt_sample, dim=0)

        total = wt_sample[-1]
        if total == 0:
            return None

        th = torch.rand() if size is None else torch.rand(size)
        idx = torch.searchsorted(wt_sample, th * total)
        result.append(idx)
    return  torch.tensor(result)


def sample(src: SparseTensor, num_neighbors: int,
           subset: Optional[torch.Tensor] = None) -> torch.Tensor:

    rowptr, col, weight = src.csr()
    rowcount = src.storage.rowcount()

    if subset is not None:
        rowcount = rowcount[subset]
        rowptr = rowptr[subset]

    if src.has_value():
        return weighted_sample(weight, rowptr, rowcount, num_neighbors)

    rand = torch.rand((rowcount.size(0), num_neighbors), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.view(-1, 1))

    return col[rand]


def sample_adj(src: SparseTensor, subset: torch.Tensor, num_neighbors: int,
               replace: bool = False) -> Tuple[SparseTensor, torch.Tensor]:

    rowptr, col, value = src.csr()

    rowptr, col, n_id, e_id = torch.ops.torch_sparse.sample_adj(
        rowptr, col, subset, num_neighbors, replace)

    if value is not None:
        value = value[e_id]

    out = SparseTensor(rowptr=rowptr, row=None, col=col, value=value,
                       sparse_sizes=(subset.size(0), n_id.size(0)),
                       is_sorted=True)

    return out, n_id


SparseTensor.sample = sample
SparseTensor.sample_adj = sample_adj
