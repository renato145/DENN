
#exp for dropout new prediction
#with p=0.3, 04, 0.6, 0.7,.8
# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="rosenbrock",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rosenbrock",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rosenbrock",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rosenbrock",method="NNdrop",replace_mech="Worst",D=30,runs=20,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3,dropout=0.2 cluster_job.sh



#frequency experiment

#freq=1
# sbatch --export=ALL,experiment="exp1",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rosenbrock",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh


#freq=4
# sbatch --export=ALL,experiment="exp1",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh



#freq=0.5
# sbatch --export=ALL,experiment="exp1",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=0.5,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=0.5,batch_size=4,nn_epochs=3 cluster_job.sh





#experiment Random replacement freq=1

# sbatch --export=ALL,experiment="exp1",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh



#nnpick experiment

# # #nnpick=1
# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=1,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=1,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=1,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=1,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh


# # #nnpick=3
# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh



# # #nnpick=5
# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=5,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=5,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=5,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=5,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh


# # # #nnpick=7
# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=7,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=7,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=7,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=7,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh


# # #nnpick=9
# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=9,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=9,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=9,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=9,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh



# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="noNNReval",replace_mech="Closest",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="noNNReval",replace_mech="Closest",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="noNNReval",replace_mech="Closest",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="noNNReval",replace_mech="Closest",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh


#freq experiment
# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNdrop",replace_mech="Worst",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNdrop",replace_mech="Worst",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNdrop",replace_mech="Worst",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNdrop",replace_mech="Worst",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNnorm",replace_mech="Worst",D=30,runs=30,frequency=4,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=4,batch_size=4,nn_epochs=3 cluster_job.sh



#sample size experiment 

# sbatch --export=ALL,experiment="exp1",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rosenbrock",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh

# sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="NNnorm",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=5,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh


#repeat nn_noreval for exp 1
sbatch --export=ALL,experiment="exp1",func_name="rosenbrock",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rosenbrock",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rosenbrock",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rosenbrock",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh

sbatch --export=ALL,experiment="exp1",func_name="sphere",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="sphere",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="sphere",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="sphere",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh

sbatch --export=ALL,experiment="exp1",func_name="rastrigin",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp2",func_name="rastrigin",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp3",func_name="rastrigin",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
# sbatch --export=ALL,experiment="exp4",func_name="rastrigin",method="noNNReval",replace_mech="Random",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=2,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh
