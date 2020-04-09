from pathlib import Path

out_file = Path('generated_run_experiments_nn.bash')
lines = []

# for replace_mech in ['Random','Worst']:
	# for nn_pick in [1,3,7,9]:
		# for exp in ['exp1','exp2','exp3','exp4']:
			# for func in ['sphere','rosenbrock','rastrigin']:
				# lines.append(f'''sbatch --export=ALL,experiment="{exp}",func_name="{func}",method="NNnorm",replace_mech="{replace_mech}",D=30,runs=30,frequency=1,max_times=100,nn_window=5,nn_nf=4,nn_pick={nn_pick},nn_sample_size=3,save="True",pbar="False",silent="False",cluster="True",nn_train_window=5,freqSave=1,batch_size=4,nn_epochs=3 cluster_job.sh\n''')

# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="noNN",frequency=1,frequency_save=1,diversity_method=None,save="True",pbar="False",silent="True",cluster="True"
	# for sample_size in [1,3,7,9]:	
for diversity_method in ['None','Rst','RI','HMu']:#'RI','Cw', 'None', 'Cwc', 'CwN', CwcN', 'None', 'CwN'
	for freq in [10]:	
		for replace_mech in ['Worst']: #'Random',
			for exp in ['exp1','exp2','exp3','exp4']: #'exp1','exp2','exp3',
				for func in ['sphere','rosenbrock','rastrigin']:
					lines.append(f'''sbatch --export=ALL,experiment="{exp}",func_name="{func}",method="NNnorm",frequency="{freq}",frequency_save="{freq}",diversity_method="{diversity_method}",scale_factor="Random",save="True",pbar="False",silent="False",cluster="True",replace_mech={replace_mech},nn_window=5,nn_nf=4,nn_pick=5,nn_sample_size=3,nn_epochs=4,nn_train_window=1,batch_size=4 cluster_job_nn.sh\n''')
with open(out_file, 'w') as f: f.writelines(lines)

print('Done!')

# sbatch --export=ALL,experiment="exp1",func_name="sphere",method="noNN",frequency=1,frequency_save=1,diversity_method=None,save="True",pbar="False",silent="True",cluster="True",replace_mech="Worst",nn_window=5,nn_nf=4,nn_pick=3,nn_sample_size=3,nn_epochs=3,nn_train_window=5,batch_size=4