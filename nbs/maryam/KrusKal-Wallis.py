from denn import *
from scipy.stats import kruskal
import scikit_posthocs as sp
import pylustrator

pylustrator.start()

path = Path('../../data/results/experiment4')

# fitness plots
no_nn = pd.read_csv(path/'no_nn_mof.csv')
nn_normal_rand = pd.read_csv(path/'nn-normal-random_mof.csv')
nn_dist_rand = pd.read_csv(path/'nn-distribution-random_mof.csv')
nn_dropout_rand= pd.read_csv(path/'nn-dropout-random_mof.csv')
labels = ['no_nn', 'nn_normal_rand', 'nn_dist_rand', 'nn_drop_rand']

x=np.array([no_nn.mof, nn_normal_rand.mof, nn_dist_rand.mof,nn_dropout_rand.mof])

stat, p = kruskal(no_nn,nn_normal_rand,nn_dist_rand,nn_dropout_rand)
pc=sp.posthoc_conover(x, p_adjust='holm', val_col='values', group_col='groups')
print('Statistics=%.3f, p=%.3f' % (stat, p))
print(pc)
heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

ax,cbar = sp.sign_plot(pc, **heatmap_args)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.show()
