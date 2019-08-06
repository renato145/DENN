from run_experiment import main
from parallel_ex import ParallelEx

def run():
    kwargs = [
        dict(experiment='exp1', func_name='rosenbrock', method='noNNReval', runs=20, frequency=1200, pbar=False),
        #dict(experiment='exp1', func_name='rastrigin', method='noNNRestart', runs=30, frequency=1200, pbar=False),
        dict(experiment='exp1', func_name='rosenbrock', method='NNnorm', replace_mech='Worst', runs=20, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='rosenbrock', method='NNdrop', replace_mech='Worst', runs=20, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='rosenbrock', method='NNnorm', replace_mech='Closest', runs=20, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='rosenbrock', method='NNdrop', replace_mech='Closest', runs=20, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='rosenbrock', method='NNnorm', replace_mech='Random', runs=20, frequency=1000, pbar=False),
        dict(experiment='exp1', func_name='rosenbrock', method='NNdrop', replace_mech='Random', runs=20, frequency=1000, pbar=False),
    ]
    ParallelEx.run_kwargs(main, kwargs, totals=30, n_cpu=None)

if __name__ == "__main__":
    run()
