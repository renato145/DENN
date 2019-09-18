#Experiment = Enum('Experiment', 'exp1 exp2 exp3 exp4')
#Method = Enum('Methods', 'noNNRestart noNNReval NNnorm NNdrop')
#Mechname:Random, Worst, Closest

# ~/anaconda3/envs/DENN/bin/python run_experiment.py exp1 NNnorm Worst --D 30 --runs 30 --frequency 1000 --max_times 100 --nn_window 5 --nn_nf 4 --nn_pick 3

#exp1
~/anaconda3/envs/DENN/bin/python run_experiment.py exp2 sphere NNdrop  Random --runs 1 --frequency 1 --nn_window 5 --nn_train_window 3 --nn_sample_size 3 --nn_epochs 3 --batch_size 8 --freq_save=1
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp1 noNNRestart  --runs 30 --frequency 1200
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp1 rosenbrock NNnorm  Worst --runs 20 --frequency 1000
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp1 rosenbrock NNdrop  Worst --runs 20 --frequency 1000
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp1 rosenbrock NNnorm  Closest --runs 30 --frequency 1000
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp1 rosenbrock NNdrop  Closest --runs 30 --frequency 1000
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp1 rosenbrock NNnorm  Random --runs 30 --frequency 1000
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp1 rosenbrock NNdrop  Random --runs 30 --frequency 1000

#exp4
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp4 noNNReval  --runs 30 --frequency 1150
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp4 noNNRestart  --runs 30 --frequency 1150
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp4 NNnorm  Worst --runs 30 --frequency 1000
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp4 NNdrop  Worst --runs 30 --frequency 1000
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp4 NNnorm  Closest --runs 30 --frequency 1000
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp4 NNdrop  Closest --runs 30 --frequency 1000
#~/anaconda3/envs/DENN/bin/python run_experiment.py exp4 NNnorm  Random --runs 30 --frequency 1000
