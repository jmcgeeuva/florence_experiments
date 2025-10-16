import torch

def test(p, r):
    unique_r, inverse_indices = torch.unique(r, return_inverse=True)
    max_p_per_group = torch.zeros_like(unique_r)
    max_p_per_group = max_p_per_group.scatter_reduce(0, inverse_indices, p, reduce='amax', include_self=False)
    p_new = max_p_per_group[inverse_indices]

    idx = torch.arange(len(r))
    last_occurrence = scatter_last = torch.zeros_like(r, dtype=torch.bool)
    scatter_last = scatter_last.to(dtype=inverse_indices.dtype).scatter(0, inverse_indices, idx.to(dtype=inverse_indices.dtype))
    mask = (idx == scatter_last[inverse_indices])

    r_new = r.clone()
    r_new[~mask] = 0
    keep_mask = mask.int()

    ap = torch.sum(p_new*keep_mask)/torch.sum(keep_mask)
    return p_new, keep_mask

import pdb; pdb.set_trace()
