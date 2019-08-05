from run_experiment import main
from parallel_ex import ParallelEx

def run():
    kwargs = [
        dict(experiment='exp1', func_name='sphere', method='noNNReval', runs=30, frequency=1150, pbar=False),
        dict(experiment='exp1', func_name='sphere', method='noNNRestart', runs=30, frequency=1150, pbar=False),
        dict(experiment='exp1', func_name='sphere', method='NNnorm', replace_mech='Worst', runs=30, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='sphere', method='NNdrop', replace_mech='Worst', runs=30, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='sphere', method='NNnorm', replace_mech='Closest', runs=30, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='sphere', method='NNdrop', replace_mech='Closest', runs=30, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='sphere', method='NNnorm', replace_mech='Random', runs=30, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='sphere', method='NNdrop', replace_mech='Random', runs=30, frequency=1000, pbar=False),
    ]
    ParallelEx.run_kwargs(main, kwargs, totals=30, n_cpu=None)

if __name__ == "__main__":
    run()
