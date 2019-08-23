from run_experiment import main
from parallel_ex import ParallelEx

def run():
    kwargs = [
       # dict(experiment='exp2', func_name='sphere', method='NNnorm', replace_mech='Random', runs=1, frequency=1000, pbar=False),
        #dict(experiment='exp2', func_name='sphere', method='NNnorm', replace_mech='Random', runs=1, frequency=1000, nn_train_window=5,
          #  nn_sample_size=3, pbar=False),
       # dict(experiment='exp2', func_name='sphere', method='NNnorm', replace_mech='Random', runs=1, frequency=1000, nn_train_window=5,
        #    nn_sample_size=5, pbar=False),

        # dict(experiment='exp3', func_name='rosenbrock', method='noNNReval', runs=1, frequency=620, pbar=False, freqSave=5000),
        # dict(experiment='exp1', func_name='rosenbrock', method='noNNRestart', runs=30, frequency=1200, pbar=False, freqSave=5000),
        
        dict(experiment='exp1', func_name='sphere', method='NNnorm', replace_mech='Worst', runs=5, frequency=5000, pbar=False, freqSave=5000),
        dict(experiment='exp1', func_name='sphere', method='NNdrop', replace_mech='Worst', runs=5, frequency=5000, pbar=False, freqSave=5000),
        dict(experiment='exp2', func_name='sphere', method='NNnorm', replace_mech='Worst', runs=5, frequency=5000, pbar=False, freqSave=5000),
        dict(experiment='exp2', func_name='sphere', method='NNdrop', replace_mech='Worst', runs=5, frequency=5000, pbar=False, freqSave=5000),
        dict(experiment='exp3', func_name='sphere', method='NNnorm', replace_mech='Worst', runs=5, frequency=5000, pbar=False, freqSave=5000),
        dict(experiment='exp3', func_name='sphere', method='NNdrop', replace_mech='Worst', runs=5, frequency=5000, pbar=False, freqSave=5000),
        dict(experiment='exp4', func_name='sphere', method='NNnorm', replace_mech='Worst', runs=5, frequency=5000, pbar=False, freqSave=5000),
        dict(experiment='exp4', func_name='sphere', method='NNdrop', replace_mech='Worst', runs=5, frequency=5000, pbar=False, freqSave=5000),
       #  dict(experiment='exp2', func_name='sphere', method='NNnorm', replace_mech='Closest', runs=20, frequency=5000, pbar=False, freqSave=5000),
       #  dict(experiment='exp2', func_name='sphere', method='NNdrop', replace_mech='Closest', runs=20, frequency=5000, pbar=False,freqSave=5000),
       #  dict(experiment='exp2', func_name='sphere', method='NNnorm', replace_mech='Random', runs=20, frequency=5000, pbar=False,freqSave=5000),
       #  dict(experiment='exp2', func_name='sphere', method='NNdrop', replace_mech='Random', runs=20, frequency=5000, pbar=False,freqSave=5000),
    ]
    ParallelEx.run_kwargs(main, kwargs, totals=30, n_cpu=None)

if __name__ == "__main__":
    run()
