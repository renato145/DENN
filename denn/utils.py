from .imports import *

__all__ = ['pick_n_but', 'get_unique']

def pick_n_but(n:int, idx:int, collection:Collection[Any]):
    'pick `n` objects from `collection` not in index `idx`'
    idxs = list(range(len(collection)))
    idxs.pop(idx)
    idxs = np.random.choice(idxs, n, replace=False)
    return [collection[i] for i in idxs]

def get_unique(x:Collection[Any]): return list(set(x))
