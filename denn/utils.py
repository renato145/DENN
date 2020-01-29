from .imports import *

__all__ = ['pick_n_but', 'get_unique', 'listify', 'ifnone', 'is_listy', 'parallel', 'camel2snake', 'SchedLin', 'SchedCos', 'SchedExp']

@jit(nopython=True)
def pick_n_but(n:int, idx:int, size:int):
    'pick `n` objects from not in index `idx`'
    idxs = list(range(size))
    idxs.pop(idx)
    idxs = np.random.choice(np.array(idxs), n, replace=False)
    return idxs

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

def parallel(func, arr:Collection, max_workers:int=None):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, cpu_count())
    if max_workers<2: results = [func(o,i) for i,o in progress_bar(enumerate(arr), total=len(arr))]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
            results = []
            for f in progress_bar(concurrent.futures.as_completed(futures), total=len(arr)): results.append(f.result())
    if any([o is not None for o in results]): return results

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name:str)->str:
    "Change `name` from camel to snake style."
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def annealer(f):
    'Decorator to make `f` return itself partially applied.'
    @functools.wraps(f)
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def SchedLin(start, end, pos): return start + pos*(end-start)
@annealer
def SchedCos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
@annealer
def SchedExp(start, end, pos): return start * (end/start) ** pos
