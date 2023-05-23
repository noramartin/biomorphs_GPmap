#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
from os.path import isfile
import sys
import numpy as np
import matplotlib.pyplot as plt
from functions_for_GP_analysis.GPproperties import is_tree, tuple_to_binary_string, random_startgene, plot_phenotype, df_data_to_dict, decimal_to_scientific_notation, approximate_upper_bound_complexity_plot
import pandas as pd
from copy import deepcopy
from functools import partial
from scipy import optimize
#from matplotlib import colors
from collections import Counter
"""
runs random walks and records discovery of a few specific shapes
"""

def population_in_flat_fitness_landscape(N, mu, genemax, g9max, g9min, phenotypes, Tmax, is_viable_phenotype, phenotypes_of_interest): #, phenotypes):
   def fitness(phenotype):
      if is_viable_phenotype(phenotype):
         return 1
      else:
         return 0  
   ##
   startgene = random_startgene(genemax, g9min, g9max)
   while not has_viable_phenotype(phenotypes[tuple(startgene)]):
      startgene = random_startgene(genemax, g9min, g9max)
   print('found initial genotype')
   phenotype_vs_discovery_times_and_individuals = {ph: [] for ph in phenotypes_of_interest}
   phenotype_vs_count_only = {}
   #use initial conditions
   P_g_new = np.zeros((N,9), dtype='int16') #population genotypes
   P_g_current =  np.tile(startgene[:], (N,1)) #use initial condition for population
   P_p_current = np.tile(phenotypes[tuple(startgene)], (N))
   P_fitness_current = np.tile(fitness(phenotypes[tuple(startgene)]), (N))
   ########start simulation
   initialisation_factor = 10
   for t in range(0, int(Tmax) + initialisation_factor * N): # thermalisation first
      #choose next generation
      rows = np.random.choice(np.arange(N), N, replace=True, p=np.divide(P_fitness_current[:],float(np.sum(P_fitness_current[:])))) #select N rows that get chosen for the next generation
      for individual in range(N):
         for pos in range(len(P_g_current[rows[individual],:])):
            P_g_new[individual,pos] = P_g_current[rows[individual],pos] #fill in genes for next generation         
            #mutate with p=mu
            mutate=np.random.choice([True, False],p=[mu, 1-mu]) #generate mutations
            if mutate:
               if P_g_new[individual,pos] == phenotypes.shape[pos] - 1: 
                  P_g_new[individual,pos] =  int(P_g_new[individual,pos] - 1)
               elif P_g_new[individual,pos]==0: 
                  P_g_new[individual,pos] = int(P_g_new[individual,pos]+1)
               else:
                  P_g_new[individual,pos] = int(np.random.choice([P_g_new[individual,pos]-1, P_g_new[individual,pos]+1]))
         #fill in phenotype and fitness
         P_p_current[individual]=phenotypes[tuple(P_g_new[individual,:])]
         P_fitness_current[individual]=fitness(int(P_p_current[individual]))
         if t >= initialisation_factor * N:
         	try:
         		phenotype_vs_count_only[P_p_current[individual]] += 1
         	except KeyError:
         		phenotype_vs_count_only[P_p_current[individual]] = 1
         if t >= initialisation_factor * N and P_p_current[individual] in phenotypes_of_interest: #and P_p_current[individual] != pheno_before_mutation
            phenotype_vs_discovery_times_and_individuals[P_p_current[individual]].append((t - initialisation_factor * N, individual))
      P_g_current[:,:] = deepcopy(np.copy(P_g_new[:,:]))
      if np.count_nonzero(P_fitness_current) == 0:
         raise RuntimeError( 'Error: Only zero-fitness individuals left in the population.')
      if t%100==0 and t!=0:
         print('log10 population diversity', np.log10(len(set(P_p_current[:]))))
         print( str(float(t - initialisation_factor * N)/Tmax *100)+'% finished')
   return phenotype_vs_discovery_times_and_individuals, phenotype_vs_count_only


################################################################################################################################################################################################
### parameters
################################################################################################################################################################################################

####################################################################################
####################################################################################
max_no_steps_discoverytime_loop = 10**5 # 10**5
no_steps_for_filename = str(max_no_steps_discoverytime_loop)[0] + '_' +  str(int(np.log10(max_no_steps_discoverytime_loop)))
N = 2 * 10**3
mu = 0.1
####################################################################################
####################################################################################
genemax = int(sys.argv[1])
g9min = int(sys.argv[2])
g9max = int(sys.argv[3])
resolution = int(sys.argv[4])
threshold = int(sys.argv[5])
type_walk = sys.argv[6] #'all' or 'tree'
string_for_saving_plot = str(genemax)+'_'+str(g9min)+'_'+str(g9max) +'_'+str(resolution) + '_'+str(threshold)
####################################################################################
####################################################################################
if type_walk == 'tree':
   filename_trees = './data_flat_landscape/which_phenotypes_are_trees' + string_for_saving_plot + '.npy'
   if not isfile(filename_trees):
      tuple_vs_int_list = np.load('./biomorphs_map_pixels/phenotypes_number_dict'+string_for_saving_plot+'.npy', allow_pickle=True).item()
      int_vs_tuple_phenotype = {intph: tuple(tupleph[:]) for tupleph, intph in tuple_vs_int_list.items()}
      int_vs_istree = np.zeros(max(int_vs_tuple_phenotype.keys()) + 1, dtype='uint16')
      for ph, tupleph in int_vs_tuple_phenotype.items():
         if is_tree(tuple_to_binary_string(tupleph, resolution)):
            int_vs_istree[ph] = 1
         else:
            int_vs_istree[ph] = 0
      np.save(filename_trees, int_vs_istree)
   else:
      int_vs_istree = np.load(filename_trees)
   def has_viable_phenotype(ph_int):
      return int_vs_istree[ph_int]
elif type_walk == 'all':
   def has_viable_phenotype(ph):
      return True
else:
   raise RuntimeError('unknown walk type')
####################################################################################
####################################################################################

phenotypes = np.load('./biomorphs_map_pixels/phenotypes'+string_for_saving_plot+'.npy')
filename_GPmapproperties_type_walk = './biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_forrandomwalk_' + type_walk+'.csv'
print('expected # mutations for phenotype with neutral set size of 2', N * mu * len(phenotypes.shape) * max_no_steps_discoverytime_loop * 2 /np.product(phenotypes.shape))
if not isfile(filename_GPmapproperties_type_walk):
   df_GPmap_properties = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'.csv')
   ph_vs_f = {ph: f for ph, f in zip(df_GPmap_properties['phenotype int'].tolist(), df_GPmap_properties['neutral set size'].tolist()) if has_viable_phenotype(ph)}
   ph_vs_arrayKC  = df_data_to_dict('array complexity BDM', df_GPmap_properties)
   phenotypes_sorted_by_f = sorted(ph_vs_f.keys(), key=ph_vs_f.get, reverse=True)
   df_to_save_with_selected_phenos = pd.DataFrame.from_dict({'phenotype int': phenotypes_sorted_by_f, 
                                                             'neutral set size': [ph_vs_f[ph] for ph in phenotypes_sorted_by_f],
                                                             'array complexity BDM': [ph_vs_arrayKC[ph] for ph in phenotypes_sorted_by_f]})
   df_to_save_with_selected_phenos.to_csv(filename_GPmapproperties_type_walk)
else:
   df_GPmap_properties = pd.read_csv(filename_GPmapproperties_type_walk)
   ph_vs_f = {ph: f for ph, f in zip(df_GPmap_properties['phenotype int'].tolist(), df_GPmap_properties['neutral set size'].tolist())}
   ph_vs_arrayKC  = df_data_to_dict('array complexity BDM', df_GPmap_properties)   
####################################################################################
log_len_rank_plot = np.log10(len(ph_vs_f))
phenotypes_sorted_by_f = sorted(ph_vs_f.keys(), key=ph_vs_f.get, reverse=True)


####################################################################################

np.random.seed(5)
####################################################################################
print('select phenotypes to focus on', flush=True)
####################################################################################

freq_cutoff = int(10**(0.1 * log_len_rank_plot))
assert freq_cutoff - 2 > 2 or (type_walk == 'tree' and freq_cutoff > 0) # otherwise not enough choices available to select two distinct phenotypes that are not the top two (these might be artefacts)
if type_walk == 'all':
   phenotype_examples = [np.random.choice(phenotypes_sorted_by_f[2:freq_cutoff]),]  #pick a simple phenotype, but not simple line or empty phenotype
else:
   phenotype_examples = [np.random.choice(phenotypes_sorted_by_f[:freq_cutoff]),]  #pick a simple phenotype
phenotype_examples.append(np.random.choice(phenotypes_sorted_by_f[int(10**(0.45 * log_len_rank_plot)): int(10**(0.55 * log_len_rank_plot))])) #pick a more complex phenotype
phenotype_examples.append(np.random.choice(phenotypes_sorted_by_f[int(10**(0.9 * log_len_rank_plot)):])) #pick a complex phenotype
###
assert len(phenotype_examples) == len(set(phenotype_examples))  
################################################################################################################################################################################################
print('randomwalk', flush=True)
################################################################################################################################################################################################
details_filename = string_for_saving_plot + 'chosen_examples' + '_'.join([str(ph) for ph in phenotype_examples]) + 'steps' + no_steps_for_filename + 'N' +str(N) + 'mu' + str(int(1000*mu))
filename_random_walk ='./data_flat_landscape/barchart_data' + details_filename + '.csv'
filename_random_walk_allphenos ='./data_flat_landscape/barchart_data' + details_filename + '_allphenos.csv'
if not (isfile(filename_random_walk) and isfile(filename_random_walk_allphenos)):
   phenotype_vs_discovery_times_and_individuals, phenotype_vs_count_only = population_in_flat_fitness_landscape(N, mu, genemax, g9max, g9min, phenotypes, max_no_steps_discoverytime_loop, has_viable_phenotype, phenotype_examples)
   phenotype_list, time_list, individual_list = zip(*[(ph, t, i) for ph in phenotype_vs_discovery_times_and_individuals for (t, i) in phenotype_vs_discovery_times_and_individuals[ph]])
   df_discoverytimes = pd.DataFrame.from_dict({'phenotype': phenotype_list, 'time': time_list, 'individual': individual_list})
   df_discoverytimes.to_csv(filename_random_walk)
   df_discoverytimes_allphenos = pd.DataFrame.from_dict({'phenotype': [p for p in phenotype_vs_count_only], 'count': [phenotype_vs_count_only[p] for p in phenotype_vs_count_only]})
   df_discoverytimes_allphenos.to_csv(filename_random_walk_allphenos)
else:
   print('load randomwalk data', flush = True)
   df_discoverytimes = pd.read_csv(filename_random_walk)
   phenotype_vs_discovery_times_and_individuals = {ph: [] for ph in phenotype_examples}
   for rowindex, row in df_discoverytimes.iterrows():
      phenotype_vs_discovery_times_and_individuals[row['phenotype']].append((row['time'], row['individual']))
   df_discoverytimes_allphenos = pd.read_csv(filename_random_walk_allphenos)
   phenotype_vs_count_only = dict(zip(df_discoverytimes_allphenos['phenotype'].tolist(), df_discoverytimes_allphenos['count'].tolist())) #{row['phenotype']: row['count'] for rowindex, row in df_discoverytimes_allphenos.iterrows()}

print('number of occurences', [len(phenotype_vs_discovery_times_and_individuals[ph]) for ph in phenotype_examples])

################################################################################################################################################################################################   
################################################################################################################################################################################################
print('plot', flush=True)
################################################################################################################################################################################################
################################################################################################################################################################################################
ph_vs_color = {ph: color for ph, color in  zip(phenotype_examples, ['orange', 'teal', 'purple'])}
################################################################################################################################################################################################
print('plots the example phenotypes in specific colous', flush=True)
################################################################################################################################################################################################
for i, ph in enumerate(phenotype_examples):
   f, ax = plt.subplots(figsize=(1.5, 1.55))
   plot_phenotype(ph, ax, phenotypes, g9min, color = ph_vs_color[ph])
   #ax.set_title('neutral set size: ' + str(ph_vs_f[ph]))
   normalised_f = ph_vs_f[ph]/float(np.prod(phenotypes.shape))
   print(str(ph) + ' - neutral set size: ' + str(ph_vs_f[ph]) + ' or if normalised' + str(normalised_f))
   f.savefig('./plots_flat_landscape/'+str(ph)+'_phenotypes_'+string_for_saving_plot+'.png', bbox_inches='tight', dpi=200)
   plt.close()
   del f, ax
   print(i, ph, 'number of times expected per generation', N * normalised_f)

################################################################################################################################################################################################
print('timeline with discovery times 2', flush=True)
################################################################################################################################################################################################
f, ax = plt.subplots(figsize=(8, 3))
for i, ph in enumerate(phenotype_examples):
   #if i == 2:
   #scale_marker = min([(np.log10(ph_vs_f[phenotype_examples[0]])/np.log10(ph_vs_f[ph]))**4, 10.0])
   #else:
   scale_marker = 0.7 * min([(np.log10(ph_vs_f[phenotype_examples[0]])/np.log10(ph_vs_f[phenotype_examples[1]]))**2, 10.0])
   if ph in phenotype_vs_discovery_times_and_individuals and len(phenotype_vs_discovery_times_and_individuals[ph]) > 0:
      if i == 2:
         time_step = 1
      else:
         time_step = max_no_steps_discoverytime_loop//1000
      phenotype_vs_discovery_times_and_individuals_toplot = [(t, i) for t, i in phenotype_vs_discovery_times_and_individuals[ph] if t%time_step == 0]
      if len(phenotype_vs_discovery_times_and_individuals_toplot) == 0:
         continue
      ax.scatter(list(zip(*phenotype_vs_discovery_times_and_individuals_toplot))[0],
                 list(zip(*phenotype_vs_discovery_times_and_individuals_toplot))[1],
                 marker = 'x', c=ph_vs_color[ph], s=10*scale_marker, alpha=[0.5, 0.6, 1][i], zorder= i, linewidths=[0.1, 0.2, 5][i])
ax.set_xlabel('number of generations ('+r'$ \times 10^4$'+')', fontsize=12)
#ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False) # from matplotlib documentation ("Centered spines with arrows")
ax.set_xticks(np.arange(max_no_steps_discoverytime_loop+1, step = max_no_steps_discoverytime_loop//5))
ax.set_xticklabels([str(int(t/10**4)) for t in np.arange(max_no_steps_discoverytime_loop+1, step = max_no_steps_discoverytime_loop//5)], fontsize=12)
ax.set_xlim(0, max_no_steps_discoverytime_loop*1.03)
ax.set_ylim(-1, N + 1)
ax.set_ylabel('individual\n(out of '+str(N) + ' individuals\nin the population)', fontsize=12)
ax.set_yticks(np.arange(N+1, step = N//4))
ax.set_yticklabels([t for t in np.arange(N+1, step = N//4)], fontsize=11)#[decimal_to_scientific_notation(t, 0, b=2) for t in np.arange(N+1, step = N//4)], fontsize=16)
f.tight_layout()
f.savefig('./plots_flat_landscape/timeline_' + details_filename + '_2.png', dpi=250)


####################################################################################
print('plot NSS with examples underneath and scatter', flush=True)
####################################################################################
number_genotypes = np.prod(phenotypes.shape)
f, ax = plt.subplots(ncols=2, figsize=(9, 2.5), gridspec_kw={'width_ratios': [2.5, 1]})
NSS_sorted=np.array(sorted([N for N in ph_vs_f.values()], reverse=True))
N_vs_rank_data = {f: (NSS_sorted >= f - 0.01).sum() for f in set(NSS_sorted)}
rank_data = [N_vs_rank_data[f] for f in NSS_sorted]
ax[0].plot(rank_data, np.divide(NSS_sorted, float(number_genotypes)), c='grey', lw=0.8, marker='o', ms=4, alpha = 0.8, mew=0.0, zorder=-1)
ax[0].set_xlim(0.8, 1.2*len(NSS_sorted))
ax[0].set_ylim(0.15/number_genotypes, 6.0 * max(NSS_sorted)/float(number_genotypes))
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel('frequency rank of each phenotype')
ax[0].set_ylabel(r'phenotypic frequency, $f_p$')
###for pheno1 (draw in green)
for i, ph in enumerate(phenotype_examples):
   rankpheno =  N_vs_rank_data[ph_vs_f[ph]]
   ax[0].scatter(rankpheno, ph_vs_f[ph]/float(number_genotypes), c=ph_vs_color[ph], edgecolor=ph_vs_color[ph], alpha=1, s=140, marker='*', zorder=0, edgecolors='white')
####
print(sum(phenotype_vs_count_only.values()),  N * max_no_steps_discoverytime_loop)
assert sum(phenotype_vs_count_only.values()) == N * max_no_steps_discoverytime_loop
number_found_list = [phenotype_vs_count_only[p] if p in phenotype_vs_count_only else 0 for p in ph_vs_f.keys()]
frequency_list = [ph_vs_f[p] for p in ph_vs_f.keys()]
norm_freq, norm_number_found = np.sum(frequency_list), 1.0 * N * max_no_steps_discoverytime_loop
assert norm_number_found == np.sum(number_found_list) or type_walk == 'tree'
assert norm_freq == number_genotypes or type_walk == 'tree'
print('fraction of all genotypes that are viable', norm_freq/number_genotypes, 'fraction of phenotypes in population that are viable', np.sum(number_found_list)/norm_number_found)
ax[1].scatter(np.divide(frequency_list, norm_freq), np.divide(number_found_list, norm_number_found), s=3, c='grey', alpha = 0.5, zorder=1)
#regression = linregress(np.log10(frequency_list), np.log10(number_found_list))
#print('regression slope is', regression.slope, 'should be 1')
for ph in phenotype_examples:
   ax[1].scatter(ph_vs_f[ph]/norm_freq, phenotype_vs_count_only[ph]/norm_number_found, c=ph_vs_color[ph], edgecolor=ph_vs_color[ph], alpha=1, s=140, marker='*', zorder=2, edgecolors='white')
ax[1].set_xscale("log")
xlims = [min(frequency_list)/norm_freq * 0.5, max(frequency_list)/norm_freq * 2]
ax[1].set_xlim(xlims[0], xlims[1])
ax[1].plot(xlims, xlims, c='k', zorder=-5)

ax[1].set_yscale("log")
ax[1].set_ylim(0.5/norm_number_found, max(number_found_list)/norm_number_found * 2)
ax[1].set_xlabel(r'phenotypic frequency $f_p$')
ax[1].set_ylabel(r'mean frequency of $p$'+ ' in\nevolving population')
f.tight_layout()
f.savefig('./plots_flat_landscape/rank_of_examples_and_barchart'+details_filename+string_for_saving_plot+'_all2.png', bbox_inches='tight', dpi=1200)



####################################################################################
print('plot NSS with examples underneath and barcharts as well - include complexity plot', flush=True)
####################################################################################
number_genotypes = np.prod(phenotypes.shape)
f, ax = plt.subplots(ncols= 3, figsize=(9.5, 2.3), gridspec_kw={'width_ratios': [2, 1, 1]})
NSS_sorted=np.array(sorted([N for N in ph_vs_f.values()], reverse=True))
N_vs_rank_data = {f: (NSS_sorted >= f - 0.01).sum() for f in set(NSS_sorted)}
rank_data = [N_vs_rank_data[f] for f in NSS_sorted]
ax[0].plot(rank_data, np.divide(NSS_sorted, float(number_genotypes)), c='grey', lw=0.8, marker='o', ms=4, alpha = 0.8, mew=0.0, zorder=-1)
ax[0].set_xlim(0.8, 1.2*len(NSS_sorted))
ax[0].set_ylim(0.15/number_genotypes, 6.0 * max(NSS_sorted)/float(number_genotypes))
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel('frequency rank of each phenotype')
ax[0].set_ylabel(r'phenotypic frequency, $f_p$')
###for pheno1 (draw in green)
for i, ph in enumerate(phenotype_examples):
   rankpheno =  N_vs_rank_data[ph_vs_f[ph]]
   ax[0].scatter(rankpheno, ph_vs_f[ph]/float(number_genotypes), c=ph_vs_color[ph], edgecolor=ph_vs_color[ph], alpha=1, s=70, marker='*', zorder=0)
   ax[1].scatter(ph_vs_arrayKC[ph], ph_vs_f[ph]/float(number_genotypes), c=ph_vs_color[ph], edgecolor=ph_vs_color[ph], alpha=1, s=70, marker='*', zorder=0)
###
frequency_list_data = np.divide([ph_vs_f[ph] for ph in phenotypes_sorted_by_f], float(np.prod(phenotypes.shape)))
complexity_list_data=[ph_vs_arrayKC[pheno] for pheno in phenotypes_sorted_by_f]

ax[1].set_ylabel('frequency')
ax[1].set_xlabel('complexity estimate')
ax[1].scatter(complexity_list_data, frequency_list_data, c='grey', edgecolor=None, alpha=0.1, s=3, zorder=-1)
ax[1].set_yscale('log')

x_list = np.linspace(min(complexity_list_data)*0.5, max(complexity_list_data)*1.5, 100)
slope, intercept = approximate_upper_bound_complexity_plot(complexity_list_data, frequency_list_data)
ax[1].plot(x_list, np.power(2, np.array(x_list)* slope + intercept), c='k')
ax[1].set_ylim(0.1*min(frequency_list_data), 1.15*max(max(np.power(2, np.array(x_list)*slope + intercept)), max(frequency_list_data)))
ax[1].set_xlim(0.75*min(complexity_list_data), 1.25*max(complexity_list_data))
####
ax[2].scatter(np.divide(frequency_list, norm_freq), np.divide(number_found_list, norm_number_found), s=3, c='grey', alpha = 0.5, zorder=1)
#regression = linregress(np.log10(frequency_list), np.log10(number_found_list))
#print('regression slope is', regression.slope, 'should be 1')
for ph in phenotype_examples:
   ax[2].scatter(ph_vs_f[ph]/norm_freq, phenotype_vs_count_only[ph]/norm_number_found, c=ph_vs_color[ph], edgecolor=ph_vs_color[ph], alpha=1, s=140, marker='*', zorder=2)
ax[2].set_xscale("log")
xlims = [min(frequency_list)/norm_freq * 0.5, max(frequency_list)/norm_freq * 2]
ax[2].set_xlim(xlims[0], xlims[1])
ax[2].plot(xlims, xlims, c='k', zorder=-5)

ax[2].set_yscale("log")
ax[2].set_ylim(0.5/norm_number_found, max(number_found_list)/norm_number_found * 2)
ax[2].set_xlabel(r'phenotypic frequency $f_p$')
ax[2].set_ylabel(r'mean frequency of $p$'+ ' in\nevolving population')
f.tight_layout()
f.savefig('./plots_flat_landscape/rank_of_examples_and_barchart'+details_filename + string_for_saving_plot+'_including_complexity2.png', bbox_inches='tight', dpi=1200)
