import numpy as np
import torch


def _calc_domination_matrix(F, _F=None, epsilon=0.0):
    """
    if G is None or len(G) == 0:
        constr = np.zeros((F.shape[0], F.shape[0]))
    else:
        # consider the constraint violation
        # CV = Problem.calc_constraint_violation(G)
        # constr = (CV < CV) * 1 + (CV > CV) * -1
        CV = Problem.calc_constraint_violation(G)[:, 0]
        constr = (CV[:, None] < CV) * 1 + (CV[:, None] > CV) * -1
    """

    if _F is None:
        _F = F

    # look at the obj for dom
    n = F.shape[0]
    m = _F.shape[0]

    L = np.repeat(F, m, axis=0)
    R = np.tile(_F, (n, 1))

    smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
    larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

    M = (
        np.logical_and(smaller, np.logical_not(larger)) * 1
        + np.logical_and(larger, np.logical_not(smaller)) * -1
    )

    # if cv equal then look at dom
    # M = constr + (constr == 0) * dom

    return M


def _calc_domination_matrix_cuda(F, device, _F=None, epsilon=0.0):
    """
    if G is None or len(G) == 0:
        constr = np.zeros((F.shape[0], F.shape[0]))
    else:
        # consider the constraint violation
        # CV = Problem.calc_constraint_violation(G)
        # constr = (CV < CV) * 1 + (CV > CV) * -1
        CV = Problem.calc_constraint_violation(G)[:, 0]
        constr = (CV[:, None] < CV) * 1 + (CV[:, None] > CV) * -1
    """
    assert device

    if _F is None:
        _F = F

    # look at the obj for dom
    n = F.size()[0]
    m = _F.size()[0]

    L = torch.repeat_interleave(input=F, repeats=m, dim=0)
    R = torch.tile(_F, (n, 1))

    smaller = torch.reshape(
        torch.any(torch.tensor(L + epsilon < R, device=device), dim=1), (n, m)
    )
    larger = torch.reshape(
        torch.any(torch.tensor(L > R + epsilon, device=device), dim=1), (n, m)
    )

    M = (
        torch.logical_and(smaller, torch.logical_not(larger)) * 1
        + torch.logical_and(larger, torch.logical_not(smaller)) * -1
    )

    # if cv equal then look at dom
    # M = constr + (constr == 0) * dom

    return M


def _fast_non_dominated_sort(F, use_cuda=False, device=None):
    # calculate the dominance matrix

    if use_cuda:
        assert device
        M = _calc_domination_matrix_cuda(F, device)
    else:
        M = _calc_domination_matrix(F)

    if use_cuda:
        n = M.size()[0]
    else:
        n = M.shape[0]

    fronts = []

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=int)

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):

        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts


def _rank_from_fronts(fronts, n, use_cuda, device):
    # create the rank array and set values

    if use_cuda:
        assert device
        rank = torch.full(n, 1e16, dtype=torch.int32, device=device)
    else:
        rank = np.full(n, 1e16, dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank


def non_dominated_sorting(
    F,
    return_rank=False,
    only_non_dominated_front=False,
    n_stop_if_ranked=None,
    use_cuda=False,
    device=None,
):

    if use_cuda:
        assert device
        F = F.type(torch.float32)
    else:
        F = F.astype(float)

    if n_stop_if_ranked is None:
        n_stop_if_ranked = int(1e8)

    fronts = _fast_non_dominated_sort(F, use_cuda=use_cuda, device=device)
    # convert to numpy array for each front and filter by n_stop_if_ranked if desired
    _fronts = []
    n_ranked = 0

    for front in fronts:

        if use_cuda:
            _fronts.append(torch.tensor(front, dtype=torch.int32, device=device))
        else:
            _fronts.append(np.array(front, dtype=int))

        # increment the n_ranked solution counter
        n_ranked += len(front)

        # stop if more than this solutions are n_ranked
        if n_ranked >= n_stop_if_ranked:
            break

    fronts = _fronts

    if only_non_dominated_front:
        return fronts[0]

    if return_rank:
        rank = _rank_from_fronts(fronts, F.shape[0], use_cuda=use_cuda, device=device)
        return fronts, rank

    return fronts
