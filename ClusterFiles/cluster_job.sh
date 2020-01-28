#!/bin/bash
#SBATCH -p skylake # partition (this is the queue your job will be added to)
#SBATCH -N 1 # number of nodes (use a single node)
#SBATCH -n 1 # number of cores (sequential job uses 1 core)
#SBATCH --time=20:00:00 # time allocation, which has the format (D-HH:MM:SS), here set to 1 hour
#SBATCH --mem=4GB # memory pool for all cores (here set to 4 GB)

# Notification configuration
#SBATCH --mail-type=END # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
#SBATCH --mail-type=FAIL # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
#SBATCH --mail-user=maryam.hasanishoreh@adelaide.edu.au # Email to which notification will be sent                                                                               

# Executing script (Example here is sequential script)
python run_experiment.py $experiment $func_name $method $frequency $frequency_save $diversity_method $scale_factor $save $pbar $silent $cluster
# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="noNN",frequency=1,frequency_save=1,diversity_method=None,scale_factor="Random",save="True",pbar="False",silent="True",cluster="True"
