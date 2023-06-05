import torch


@torch.jit.script
def find_greedy_nearest_indices(weight: torch.Tensor, use_abs: bool = False):
    print("use_abs", use_abs)
    ordered_unit_weight_t = weight.detach().t().clone()

    ordered_unit_weight_t /= ordered_unit_weight_t.norm(p=2, dim=-1, keepdim=True)
    distance_matrix = ordered_unit_weight_t @ ordered_unit_weight_t.T

    if use_abs:
        distance_matrix = abs(distance_matrix)
    permutation = torch.arange(len(ordered_unit_weight_t), device=weight.device)
    for dim_i in range(len(ordered_unit_weight_t) - 2):
        nearest_dim_i = (dim_i + 1) + distance_matrix[dim_i, dim_i + 1 :].argmax()
        next_dim_i = torch.full_like(nearest_dim_i, dim_i + 1)
        index_pair = torch.stack([next_dim_i, nearest_dim_i])
        swapped_index_pair = torch.stack([nearest_dim_i, next_dim_i])
        ordered_unit_weight_t[index_pair] = ordered_unit_weight_t[swapped_index_pair]
        distance_matrix[index_pair] = distance_matrix[swapped_index_pair]
        distance_matrix[:, index_pair] = distance_matrix[:, swapped_index_pair]
        permutation[index_pair] = permutation[swapped_index_pair]
    return permutation


def get_permutation_order(H: torch.Tensor, W: torch.Tensor, permutation_order: str = "identity", use_abs: bool = False):
    """
    Permutation order for layer weights.
    :param H: Hessian of Weights
    :param W: Layer weights
    :param permutation_order: which permutation order to use default: identity, act_order,nearest
    :return: permutation order 1d int tensor
    """

    if permutation_order == "spearman":
        w_rank = W.argsort(dim=0).argsort(dim=0).float()
        w_rank = w_rank - w_rank.mean(dim=0, keepdim=True)
        perm = find_greedy_nearest_indices(w_rank, use_abs)
    elif permutation_order == "act_order":
        perm = torch.argsort(torch.diag(H), descending=True)
    elif permutation_order == "identity":
        perm = torch.arange(H.shape[0], device=H.device)
    elif isinstance(permutation_order, torch.Tensor):
        return permutation_order  # user-defined
    else:
        raise ValueError(f"Unknown permutation order name: {permutation_order}")
    return perm
