from .imports import *

__all__ = ['get_data', 'get_heatmap_data','PATH_RESULTS','labels_order']

labels_order = ['noNN_RI', 'NN_RI','noNN_HMu', 'NN_HMu', 'noNN_CwN', 'NN_CwN', 'noNN_No', 
                 'NN_No','noNN_Rst', 'NN_Rst']
PATH_RESULTS = Path(f'../../data/cluster_results')
pat = re.compile('.*/(exp\d)/(\w*)/nonn/freq([0-9\.]+)div(\w+)/(\w+)_\w+.csv')
decode_keys = ['experiment','function','freq','div','method']
nn_pat = re.compile('.*/(exp\d)/(\w*)/nn/freq([0-9\.]+)nn_w(\d+)nn_p(\d+)\w+nn_tw(\d+)\w+div([A-Za-z]+)/(\w+)_(\w+)_\w+.csv')
nn_decode_keys = ['experiment','function','freq','nnw','nnp','nntw','div','method','replace_mech']#,

def get_files(m): return list(PATH_RESULTS.glob(f'**/nonn/**/*{m}.csv'))
def get_nn_files(m): return list(PATH_RESULTS.glob(f'**/nn/**/*{m}.csv'))

def read_csv(f,m):
    df = pd.read_csv(f)
    df = df.mean().to_frame().T
    for k,v in zip(decode_keys,pat.search(str(f)).groups()): df[k] = v
    df['freq'] = df['freq'].astype(float)
    df['method'] = df['method'] + '_' + df['div']
    df.drop('div', axis=1, inplace=True)
    df['isnn']=False
    return df

def read_nn_csv(f,m):
    df = pd.read_csv(f)
    df = df.mean().to_frame().T
    for k,v in zip(nn_decode_keys,nn_pat.search(str(f)).groups()): df[k] = v
    df['freq'] = df['freq'].astype(float)
    df['method'] = df['method'] + '_' + df['replace_mech'] + '_' + df['div']
    df['method'] = df['method'].str.replace('NNnorm_Worst', 'NN')
    df['method'] = df['method'].str.replace('NNconv_Worst', 'NNconv')
    df['isnn']=True
    df.drop(['replace_mech','div'], axis=1, inplace=True)
    return df

def get_data(m, normalize=False):
    '''
    - m: the performance measure.
    '''
    files = get_files(m)
    nn_files = get_nn_files(m)
    nn_data = pd.concat([read_nn_csv(f,m) for f in nn_files])
    nonn_data = pd.concat([read_csv(f,m) for f in files])
    data = pd.concat([nn_data , nonn_data])
    if normalize:
        data_norm = (data.groupby(['experiment','function','freq','method'])[m.upper()].mean().reset_index()
                        .groupby(['experiment','function'])[m.upper()].min().reset_index()
                        .rename({m.upper():m.upper()+'_norm'}, axis=1))
        data = data.merge(data_norm, 'left')
        data[m.upper()+'_norm'] = data[m.upper()] / data[m.upper()+'_norm']
    
    return data.reset_index(drop=True)


def get_heatmap_data(m,df):
    df.function = df.function.str.title()
    df_pivot = df.pivot_table(index=['experiment','function','freq'], columns=['method'], values=[m])[m]
    return df_pivot