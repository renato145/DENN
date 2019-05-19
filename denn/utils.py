from .imports import *

__all__ = ['pick_n_but', 'get_unique', 'listify', 'ifnone', 'is_listy','camel2snake']

def pick_n_but(n:int, idx:int, collection:Collection[Any]):
    'pick `n` objects from `collection` not in index `idx`'
    idxs = list(range(len(collection)))
    idxs.pop(idx)
    idxs = np.random.choice(idxs, n, replace=False)
    return [collection[i] for i in idxs]

def get_unique(x:Collection[Any]): return list(set(x))

def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)

def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def is_listy(x:Any)->bool: return isinstance(x, (tuple,list))

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name:str)->str:
    "Change `name` from camel to snake style."
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

