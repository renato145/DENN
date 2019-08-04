from run_experiment import main
from parallel_ex import ParallelEx

def run():
    kwargs = [
        dict(experiment='exp1', method='noNNReval', runs=4, frequency=100, pbar=False),
        dict(experiment='exp2', method='noNNReval', runs=4, frequency=100, pbar=False),
        dict(experiment='exp3', method='noNNReval', runs=4, frequency=100, pbar=False),
        dict(experiment='exp4', method='noNNReval', runs=4, frequency=100, pbar=False),
    ]
    ParallelEx.run_kwargs(main, kwargs, totals=4, n_cpu=2)

if __name__ == "__main__":
    run()