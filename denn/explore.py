from .imports import *
from .utils import *

__all__ = ['ExperimentData', 'GroupData']

EXP_PATH = Path('../../data/cluster_results')
NNParams = ['freq','nn_p','div']
NoNNParams =  ['freq','div',]
GroupParams = ['experiment', 'function'] + NNParams

NNPat = re.compile('.*/(exp\d)/(\w*)/nn/freq([0-9\.]+)\w+nn_p(\d+)\w+div([A-Za-z]+)')
NNKeys = ['experiment','function'] + NNParams
NoNNPat = re.compile('.*/(exp\d)/(\w*)/nonn/freq([0-9\.]+)div(\w+)')
NoNNKeys = ['experiment','function'] + NoNNParams

MetricsFiles = ['fitness', 'nfe', 'sr', 'mof', 'sumcv', 'arr']

def custom_formatwarning(msg, *args, **kwargs): return str(msg) + '\n'
warnings.formatwarning = custom_formatwarning

def get_files(isnn:bool)->Paths:
    files = EXP_PATH.glob(f'*/*/{"nn" if isnn else "nonn"}/*')
    return [f for f in files if f.is_dir()]

def decode_file(file:PathOrStr, isnn:bool)->dict:
    if isnn: pat,keys = NNPat,NNKeys
    else    : pat,keys = NoNNPat,NoNNKeys
    return dict(zip(keys,pat.search(str(file)).groups()))

def decode_files(files:Paths, isnn:bool)->pd.DataFrame:
    return pd.DataFrame([decode_file(f,isnn) for f in files])

def ask_user(objects:Strings, title:Optional[str]=None)->str:
    t = '' if title is None else f' ({title})'
    msg = f'Select one{t}:\n'
    for i,o in enumerate(objects): msg += f'({i}) {o}\n'
    return objects[int(input(msg))]

class ExperimentData():
    def __init__(self, experiment:str, function:str, isnn:bool, **kwargs:Any):
        self.experiment,self.function,self.isnn = experiment,function,isnn
        self.params = NNParams if isnn else NoNNParams
        self._cache = None
        for k in self.params:
            value = kwargs[k] if k in kwargs else self._query_param(k)
            setattr(self, k, value)

        self._validate()

    def __repr__(self)->str:
        out = f'{self.__class__.__name__}(\n'
        for k in ['experiment','function','isnn']+self.params:
            out += f'  {k}={getattr(self,k)}\n'

        out += f'  path={self.path}\n'
        out += ')'
        return out

    @classmethod
    def from_path(self, path:PathOrStr):
        raise NotImplementedError

    def _query_param(self, param:str)->str:
        if self._cache is None: self._cache = decode_files(get_files(self.isnn), self.isnn)
        value = ask_user(sorted(self._cache[param].unique()), param)
        self._cache = self._cache.query(f'{param} == {value!r}')
        return value

    @property
    def path(self)->Path:
        path = EXP_PATH / f'{self.experiment}/{self.function}'
        if self.isnn:
            path /= f'nn/freq{self.freq}nn_w5nn_p{self.nn_p}nn_s3nn_tw5nn_bs4nn_epoch3div{self.div}/'
        else:
            path /= f'nonn/freq{self.freq}div{self.div}/'

        return path

    def _validate(self)->None:
        for metric in MetricsFiles:
            files = [o for o in self.path.ls() if metric in o.name]
            n = len(files)
            if  n==0: warnings.warn(f'No metric file for: {metric!r}')
            elif n>1:
                files_msg = '\n'.join([str(f) for f in files])
                warnings.warn(f'Too many files for metric: {metric!r}\n{files_msg}')

    def read_metric(self, metric:Optional[str]=None)->pd.DataFrame:
        if metric is None: metric = ask_user(MetricsFiles)
        assert metric in MetricsFiles, f'Invalid metric: {metric!r}\nAvailable ones are: {MetricsFiles}'
        files = [o for o in self.path.ls() if metric in o.name]
        assert len(files)==1, f'More files than expected check:\n{files}'
        return pd.read_csv(files[0])

GroupMode = Enum('GroupMode', 'NN NoNN Both')

class GroupData():
    def __init__(self, **kwargs:Any):
        self.set_params = kwargs
        if 'isnn' in self.set_params:
            self._mode = GroupMode.NN if self.set_params['isnn'] else GroupMode.NoNN
            self.set_params['isnn'] = 'nn' if self.set_params['isnn'] else 'nonn'
        else:
            self._mode = GroupMode.Both

        self.query_params = [k for k in GroupParams if k not in self.set_params]
        self.experiments = []
        self._refresh()

    def __getitem__(self, idx:int)->ExperimentData: return self.experiments[idx]
    def __setitem__(self, idx:int, val:ExperimentData)->None: self.experiments[idx] = val
    def __len__(self)->int: return len(self.experiments)
    def __iter__(self): return iter(self.experiments)

    def __repr__(self)->str:
        out = f'{self.__class__.__name__}(\n'
        for k,v in self.set_params.items(): out += f'  {k}={v}\n'
        out += f'  query params={self.query_params}\n'
        out += f') [{len(self)} elements]'
        return out

    def _refresh(self)->None:
        if len(self.query_params) == 0: return
        data = []
        experiments = []
        if self._mode != GroupMode.NN:
            nonn_data = decode_files(get_files(False), False).assign(nn_p=float('nan'), isnn=False)
            for k,v in self.set_params.items():
                if k not in ['nn_p', 'isnn']: nonn_data = nonn_data.query(f'{k} == {str(v)!r}')

            data.append(nonn_data)
            for i,row in nonn_data.iterrows():
                experiments.append(ExperimentData(**row.to_dict()))

        if self._mode != GroupMode.NoNN:
            nn_data = decode_files(get_files(True), True).assign(isnn=True)
            for k,v in self.set_params.items():
                if k != 'isnn': nn_data = nn_data.query(f'{k} == {str(v)!r}')

            data.append(nn_data)
            for i,row in nn_data.iterrows():
                experiments.append(ExperimentData(**row.to_dict()))

        self.df = pd.concat(data, ignore_index=True)
        self.experiments = experiments

