#!/usr/bin/env python3
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import definition_special_genotypes_phenotypes as param
import sys
from functions_for_GP_analysis.GPproperties import df_data_to_dict
import pandas as pd

min_number_fixations=10 #minimum number of sucessful fixation for a data point to be included
#######################################################################################################################################
##input: population properties
#######################################################################################################################################
s_list = [s1 for s1 in param.s1_list]
print(len(s_list))
N = param.N
mu = param.mu
Tmax = 10**8 #no of generations after which classed as failed
no_runs = param.no_runs
factor_initialisation = param.factor_initialisation
#######################################################################################################################################
##input: phenotypes to focus on and genotype denoting initial NC
#######################################################################################################################################
p0, p1, p2, startgene = param.p0, param.p1, param.p2, param.startgene_ind
str_startgene=''.join([str(c) for c in startgene])
print( 'p0,p1 and p2: '+str(p0)+', '+str(p1)+', '+str(p2)+', ')
#######################################################################################################################################
##input: GP map
#######################################################################################################################################
genemax = int(sys.argv[1])
g9min = int(sys.argv[2])
g9max = int(sys.argv[3])
resolution = int(sys.argv[4])
threshold = int(sys.argv[5])
string_for_saving_plot=str(genemax)+'_'+str(g9min)+'_'+str(g9max) +'_'+str(resolution) + '_'+str(threshold)
df_global = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_neutral_sets.csv')
neutral_set_size_dict  = df_data_to_dict('neutral set size', df_global)
KC_dict  = df_data_to_dict('array complexity BDM', df_global)
del df_global
print('f1, f2', neutral_set_size_dict[p1], neutral_set_size_dict[p2])
print('KC0, KC1, KC2', KC_dict[p0], KC_dict[p1], KC_dict[p2])
#######################################################################################################################################
## collect data
#######################################################################################################################################
s1_list = s_list[:-1]
s2_list = s_list[1:]
fixation_prob_matrix=np.zeros((len(s1_list), len(s2_list)))
p2_found_prob_matrix=np.zeros((len(s1_list), len(s2_list)))
for s1index, s1 in enumerate(s1_list):
  for s2index, s2 in enumerate(s2_list):
      if s2 > s1:
         string_pop_param = 'N'+str(int(N))+'_logTmax'+str(int(np.log10(Tmax)))+'_mu'+str(int(10**5 * mu))+'s1'+str(int(10**4 * s1))+'s2'+str(int(10**4 * s2))+'runs_'+str(int(no_runs)) + '_' + '_'.join([str(p) for p in [factor_initialisation, p0, p1, p2]])
         try:
            fixation_list=np.load( './data/biomorphs_fixation_'+string_pop_param + string_for_saving_plot+'.npy')
            p1_found_list=np.load( './data/biomorphs_p1found_'+string_pop_param + string_for_saving_plot+'.npy')
            p2_found_list=np.load( './data/biomorphs_p2found_'+string_pop_param + string_for_saving_plot+'.npy')
            #discovery_time_list = np.load('./data/biomorphs_discoverytimes_'+string_pop_param + string_for_saving_plot+'.npy')
            if len([fix for fix in fixation_list if fix > 0.1]) >= min_number_fixations:
               print( 'fraction of successful fixations: ', len([fix for fix in fixation_list if fix>0.1])/float(len(fixation_list)))
               fixation_prob_matrix[s1index, s2index]=len([fix for fix in fixation_list if int(fix) == 2])/float(len([fix for fix in fixation_list if fix>0.1]))
               p2_found_prob_matrix[s1index, s2index]=len([i for i in range(len(fixation_list)) if fixation_list[i]>0.1 and p2_found_list[i]>0.5])/float(len([i for i in range(len(fixation_list)) if fixation_list[i]>0.1]))
            del fixation_list, p1_found_list, p2_found_list
         except IOError:
            print('no file found for s1, s2:', s1, s2)

#######################################################################################################################################
## plot
#######################################################################################################################################

def triangular_shape_matshow(s1_list, s2_list, axi):
   for i in range(len(s1_list)):
      axi.plot([i - 0.5, i + 0.5], [i + 0.5, i + 0.5], c='k', zorder=1)
      axi.plot([i - 0.5, i - 0.5], [i - 0.5, i + 0.5], c='k', zorder=1)
   axi.plot([- 0.5, len(s1_list) - 0.5], [- 0.5, - 0.5], c='k', zorder=1)
   axi.plot([len(s1_list) - 0.5, len(s1_list) - 0.5], [- 0.5, len(s1_list) - 0.5], c='k', zorder=1)
   axi.spines['bottom'].set_visible(False)
   axi.spines['top'].set_visible(False)
   axi.spines['left'].set_visible(False)
   axi.spines['right'].set_visible(False)
   axi.set_xlim(- 0.505, len(s1_list) - 0.5 + 0.05)
   axi.set_ylim(len(s1_list) - 0.5 + 0.05, - 0.505)


f, ax = plt.subplots(ncols=2, figsize=(6.8, 2.9))
cax1=ax[0].matshow(p2_found_prob_matrix, cmap=plt.cm.Greens, zorder = 0, vmin=min([p for p in p2_found_prob_matrix.flat if p > 0.001]) * 0.5, vmax=max([p for p in p2_found_prob_matrix.flat if p > 0.001]))
plt.colorbar(cax1, ax=ax[0])
if len(s1_list) < 10:
   ax[0].set_yticks(np.arange(len(s1_list)))
   ax[0].set_yticklabels([str(round(s, 2)) for s in s1_list])
   ax[0].set_xticks(np.arange(len(s2_list)))
   ax[0].set_xticklabels([str(round(s, 2)) for s in s2_list], rotation='vertical')
else:
   ax[0].set_yticks(np.arange(len(s1_list), step=2))
   ax[0].set_yticklabels([str(round(s, 2)) for i, s in enumerate(s1_list) if i%2 == 0])
   ax[0].set_xticks(np.arange(len(s2_list), step=2))
   ax[0].set_xticklabels([str(round(s, 2)) for i, s in enumerate(s2_list) if i%2 == 0], rotation='vertical')   
ax[0].set_ylabel(r'$s_{1}$ selective advantage'+'\n'+r'of $p_1$ over $p_0$')
ax[0].set_xlabel(r'$s_{2}$ selective advantage'+'\n'+r'of $p_2$ over $p_0$')
triangular_shape_matshow(s1_list, s2_list, ax[0])
ax[0].xaxis.set_ticks_position('bottom')
ax[0].set_title(r'discovery probability of $p_{2}$')
cax2=ax[1].matshow(fixation_prob_matrix, cmap=plt.cm.Blues, vmin=min([p for p in fixation_prob_matrix.flat if p > 0.001]) * 0.5, vmax=max([p for p in fixation_prob_matrix.flat if p > 0.001]))
plt.colorbar(cax2, ax=ax[1])
if len(s1_list) < 10:
   ax[1].set_yticks(np.arange(len(s1_list)))
   ax[1].set_yticklabels([str(round(s, 2)) for s in s1_list])
   ax[1].set_xticks(np.arange(len(s2_list)))
   ax[1].set_xticklabels([str(round(s, 2)) for s in s2_list], rotation='vertical')
else:
   ax[1].set_yticks(np.arange(len(s1_list), step=2))
   ax[1].set_yticklabels([str(round(s, 2)) for i, s in enumerate(s1_list) if i%2 == 0])
   ax[1].set_xticks(np.arange(len(s2_list), step=2))
   ax[1].set_xticklabels([str(round(s, 2)) for i, s in enumerate(s2_list) if i%2 == 0], rotation='vertical')   
ax[1].set_title(r'fixation probability of $p_{2}$')
ax[1].set_ylabel(r'$s_{1}$ selective advantage'+'\n'+r'of $p_1$ over $p_0$')
ax[1].set_xlabel(r'$s_{2}$ selective advantage'+'\n'+r'of $p_2$ over $p_0$')
triangular_shape_matshow(s1_list, s2_list, ax[1])
ax[1].xaxis.set_ticks_position('bottom')
for i in range(2):
   ax[i].text(-0.17, 1.15, ['A', 'B'][i], transform=ax[i].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
f.tight_layout()
f.savefig('./plots_arrival_frequent/fixation_2Dplot_N'+str(N)+'_logTmax'+str(int(np.log10(Tmax)))+'_mu'+str(int(10**5 * mu))+'runs'+str(no_runs)+'.eps', bbox_inches='tight')
