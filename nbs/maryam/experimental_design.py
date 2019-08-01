import fire
from denn import *
from fastai.callbacks import *

Experiment = Enum('Experiment', '')
Methods = Enum('Methods', 'NoNN NNRandom NNDist')

def main(experiment, method, replace_mech, D, runs, frequency):
    pass

if __name__ == '__main__':
    fire.Fire(main)