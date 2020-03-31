from .imports import *
from .utils import *
import seaborn as sns

__all__ = ['ExperimentData', 'GroupData']

EXP_PATH = Path('../../data/cluster_results')
Params = ['freq','div']
GroupParams = ['experiment', 'function'] + Params

NNPat = re.compile('.*/(exp\d)/(\w*)/nn/freq([0-9\.]+)\w+nn_p\d+\w+div([A-Za-z]+)')
NNKeys = ['experiment','function'] + Params
NoNNPat = re.compile('.*/(exp\d)/(\w*)/nonn/freq([0-9\.]+)div(\w+)')
NoNNKeys = ['experiment','function'] + Params

MetricsFiles = ('fitness', 'nfe', 'sr', 'mof', 'sumcv', 'arr', 'nn_errors')
ExperimentParams = ('experiment','function','isnn','freq','div')

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

def metric_have_time(df:pd.DataFrame)->bool: return df.shape[1]!=1

class ExperimentData():
    def __init__(self, experiment:str, function:str, isnn:bool, **kwargs:Any):
        self.experiment,self.function,self.isnn = experiment,function,isnn
        self._cache = None
        for k in Params:
            value = kwargs[k] if k in kwargs else self._query_param(k)
            setattr(self, k, value)

        self._validate()

    @property
    def params(self)->dict: return {k:getattr(self,k) for k in ExperimentParams}

    def __repr__(self)->str:
        out = f'{self.__class__.__name__}(\n'
        for k,v in self.params.items(): out += f'  {k}={v}\n'
        out += f'  path={self.path}\n'
        out += ')'
        return out

    @classmethod
    def from_path(cls, path:PathOrStr):
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
            path /= f'nn/freq{self.freq}nn_w5nn_p3nn_s3nn_tw5nn_bs4nn_epoch3div{self.div}/'
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

    def _plot_time_metric(self, df:pd.DataFrame, ax:plt.Axes, **kwargs:Any)->None:
        values = df.mean(axis=0)
        ax.plot(values, **kwargs)

    def _plot_notime_metric(self, df:pd.DataFrame, ax:plt.Axes, **kwargs:Any)->None:
        raise NotImplementedError

    def plot_metric(self, metric:Optional[str]=None, ax:Optional[plt.Axes]=None, figsize:tuple=(6,4), **kwargs:Any)->plt.Axes:
        if ax is None: fig,ax = plt.subplots(figsize=figsize)
        df = self.read_metric(metric)
        have_time = metric_have_time(df)
        if have_time: self._plot_time_metric(df, ax, **kwargs)
        else        : self._plot_notime_metric(df, ax, **kwargs)
        return ax

GroupMode = Enum('GroupMode', 'NN NoNN Both')

class GroupData():
    def __init__(self, experiments:Collection[ExperimentData], set_params:Optional[dict]=None, query_params:Optional[Strings]=None):
        self.experiments,self.set_params,self.query_params = experiments,set_params,query_params
        self.df = pd.DataFrame([o.params for o in experiments])

    @classmethod
    def from_query(cls, **kwargs:Any):
        set_params = kwargs
        if 'isnn' in set_params:
            mode = GroupMode.NN if set_params['isnn'] else GroupMode.NoNN
            set_params['isnn'] = 'nn' if set_params['isnn'] else 'nonn'
        else:
            mode = GroupMode.Both

        query_params = [k for k in GroupParams if k not in set_params]
        assert len(query_params) > 0, 'No parameters to query...'

        data = []
        experiments = []
        if mode != GroupMode.NN:
            nonn_data = decode_files(get_files(isnn=False), False).assign(nn_p=float('nan'), isnn=False)
            for k,v in set_params.items():
                if k not in ['nn_p', 'isnn']: nonn_data = nonn_data.query(f'{k} == {str(v)!r}')

            data.append(nonn_data)
            for i,row in nonn_data.iterrows():
                experiments.append(ExperimentData(**row.to_dict()))

        if mode != GroupMode.NoNN:
            nn_data = decode_files(get_files(isnn=True), True).assign(isnn=True)
            for k,v in set_params.items():
                if k != 'isnn': nn_data = nn_data.query(f'{k} == {str(v)!r}')

            data.append(nn_data)
            for i,row in nn_data.iterrows():
                experiments.append(ExperimentData(**row.to_dict()))

        df = pd.concat(data, ignore_index=True)
        return cls(experiments, set_params, query_params)

    def __getitem__(self, idx:int)->ExperimentData: return self.experiments[idx]
    def __setitem__(self, idx:int, val:ExperimentData)->None: self.experiments[idx] = val
    def __len__(self)->int: return len(self.experiments)
    def __iter__(self): return iter(self.experiments)

    def __repr__(self)->str:
        out = f'{self.__class__.__name__}(\n'
        if self.set_params is not None:
            for k,v in self.set_params.items(): out += f'  {k}={v}\n'

        out += f'  query params={self.query_params}\n'
        out += f') [{len(self)} elements]'
        return out

    def groupby(self, col:Optional[str]=None)->Collection['GroupData']:
        valid_cols = self.df.columns
        if col is None: col = ask_user(valid_cols)
        assert col in valid_cols, f'Invalid column: {col!r}\nAvailable ones are: {valid_cols}'
        out = {val:self.__class__([self[i] for i in df.index]) for val,df in self.df.groupby(col)}
        return out

    def _plot_all(self, metric:Optional[str]=None, ax:Optional[plt.Axes]=None, figsize:tuple=(6,4), **kwargs:Any)->plt.Axes:
        if metric is None: metric = ask_user(MetricsFiles)
        assert metric in MetricsFiles, f'Invalid metric: {metric!r}\nAvailable ones are: {MetricsFiles}'
        if ax is None: fig,ax = plt.subplots(figsize=figsize)
        for o in self: o.plot_metric(metric, ax=ax, **kwargs)
        return ax

    def plot_metric(self, metric:Optional[str]=None, row:Optional[str]=None, col:Optional[str]=None, axs:Optional[plt.Axes]=None, figsize:tuple=(6,4),
                    **kwargs:Any)->plt.Axes:
        if metric is None: metric = ask_user(MetricsFiles)
        assert metric in MetricsFiles, f'Invalid metric: {metric!r}\nAvailable ones are: {MetricsFiles}'
        # Check groups
        groups = {'all':self} if row is None else self.groupby(row)
        nrows = 1 if row is None else len(groups)
        if col is None:
            groups = {k:{'all':v} for k,v in groups.items()}
            ncols = 1
        else:
            groups = {k:v.groupby(col) for k,v in groups.items()}
            ncols = np.max([len(o) for o in groups.values()])

        if axs is None: fig,axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for (row_name,row_group),row_axs in zip(groups.items(),axs):
            for (col_name,col_group),ax in zip(row_group.items(),row_axs):
                col_group._plot_all(metric, ax=ax, **kwargs)

        return axs

