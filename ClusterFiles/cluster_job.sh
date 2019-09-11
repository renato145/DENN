#!/bin/bash
#SBATCH -p batch # partition (this is the queue your job will be added to)
#SBATCH -N 1 # number of nodes (use a single node)
#SBATCH -n 1 # number of cores (sequential job uses 1 core)
#SBATCH --time=04:00:00 # time allocation, which has the format (D-HH:MM:SS), here set to 1 hour
#SBATCH --mem=4GB # memory pool for all cores (here set to 4 GB)

# Notification configuration
#SBATCH --mail-type=END # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
#SBATCH --mail-type=FAIL # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
#SBATCH --mail-user=maryam.hasanishoreh@adelaide.edu.au # Email to which notification will be sent                                                                               

# Executing script (Example here is sequential script)
python run_experiment.py $experiment $func_name $method $replace_mech $D $runs $frequency $max_times $nn_window $nn_nf $nn_pick $nn_sample_size $save $pbar $silent $cluster $nn_train_window $freqSave $batch_size $nn_epochs
# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1000,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=1,save="True",pbar="False",silent="False",cluster="True",nn_train_window=None,freqSave=500,batch_size=4,nn_epochs=10 cluster_job.sh

