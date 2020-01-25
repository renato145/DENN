#!/bin/bash
#SBATCH -p skylake # partition (this is the queue your job will be added to)
#SBATCH -N 1 # number of nodes (use a single node)
#SBATCH -n 1 # number of cores (sequential job uses 1 core)
#SBATCH --time=05:00:00 # time allocation, which has the format (D-HH:MM:SS), here set to 1 hour
#SBATCH --mem=4GB # memory pool for all cores (here set to 4 GB)

# Notification configuration
#SBATCH --mail-type=END # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
#SBATCH --mail-type=FAIL # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
#SBATCH --mail-user=maryam.hasanishoreh@adelaide.edu.au # Email to which notification will be sent                                                                               

# Executing script (Example here is sequential script)
python run_experiment.py $experiment $func_name $method $frequency $frequency_save $diversity_method $scale_factor $save $pbar $silent $cluster $replace_mech $nn_window $nn_nf $nn_pick $nn_sample_size $nn_epochs $nn_train_window $batch_size
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNnorm",frequency="0.5",frequency_save="0.5",diversity_method="None",scale_factor="Random",save="True",pbar="False",silent="False",cluster="True",replace_mech=Worst,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,nn_epochs=3,nn_train_window=5,batch_size=4 cluster_job_nn.sh
