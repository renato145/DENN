
#
# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNdrop",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=3,freqSave=1,batch_size=8,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=3,freqSave=1,batch_size=8,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNdrop",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=3,freqSave=1,batch_size=8,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=3,freqSave=1,batch_size=8,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNdrop",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=3,freqSave=1,batch_size=8,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=3,freqSave=1,batch_size=8,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNdrop",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=3,freqSave=1,batch_size=8,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=3,freqSave=1,batch_size=8,nn_epochs=3 cluster_job.sh



#nnpick=1
sbatch --export=ALL,experiment="exp1",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=1,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
sbatch --export=ALL,experiment="exp2",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=1,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
sbatch --export=ALL,experiment="exp3",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=1,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
sbatch --export=ALL,experiment="exp4",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=1,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh


#nnpick=5
sbatch --export=ALL,experiment="exp1",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=5,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
sbatch --export=ALL,experiment="exp2",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=5,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
sbatch --export=ALL,experiment="exp3",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=5,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
sbatch --export=ALL,experiment="exp4",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=5,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh




# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="noNNReval",replace_mech="Closest",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="noNNReval",replace_mech="Closest",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="noNNReval",replace_mech="Closest",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="noNNReval",replace_mech="Closest",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
