#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys
from functions_for_GP_analysis.GPproperties import *
import pandas as pd
from os.path import isfile
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from scipy.stats import pearsonr

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def linlin2Dhist(x_list, y_list, ax, f, norm='linear', binsx=50, binsy=50):
   #cmap = plt.cm.Blues
   cmap = truncate_colormap('Blues', minval=0.25, maxval=1.0, n=100) #cmr.get_sub_map('Blues', 0.2, 1)
   cmap.set_under('white')
   histcounts, binsx, binsy = np.histogram2d(x_list, y_list, bins=(binsx, binsy))
   if norm == 'linear':
      cm = ax.pcolormesh(binsx, binsy, histcounts.T, vmin=0.1, cmap=cmap)#, norm=mcolors.PowerNorm(0.5))
   elif norm == 'log':
      cm = ax.pcolormesh(binsx, binsy, histcounts.T, cmap=cmap, norm=mcolors.LogNorm(vmin=1, vmax=np.max(histcounts)))#, norm=mcolors.PowerNorm(0.5))
   cb = f.colorbar(cm, ax=ax)
   cb.set_label('count') 


def linlog2Dhist(x_list, y_list, ax, f, norm='log'):
   # from PhD thesis
   x_positive, y_positive = zip(*[xy for xy in zip(x_list, y_list) if xy[1] > 0 and not (np.isnan(xy[0]) or np.isnan(xy[1]))])
   ylims = 0.5*min(y_positive), 2*max(y_positive)
   #cmap = plt.cm.Blues
   cmap = truncate_colormap('Blues', minval=0.25, maxval=1.0, n=100) #cmr.get_sub_map('Blues', 0.2, 1)
   cmap.set_under('white')
   binsy = np.power(10, np.linspace(np.log10(ylims[0]), np.log10(ylims[1]), 50))
   xlims = (min(x_positive) - 0.05 * (max(x_positive) - min(x_positive)), max(x_positive) + 0.05 * (max(x_positive) - min(x_positive)))
   binsx = np.linspace(xlims[0], xlims[1], 50)
   histcounts = np.histogram2d(x_positive, y_positive, bins=(binsx, binsy))[0]
   if norm == 'linear':
      cm = ax.pcolormesh(binsx, binsy, histcounts.T, vmin=0.1, cmap=cmap)#, norm=mcolors.PowerNorm(0.5))
   elif norm == 'log':
      cm = ax.pcolormesh(binsx, binsy, histcounts.T, cmap=cmap, norm=mcolors.LogNorm(vmin=1, vmax=np.max(histcounts)))#, norm=mcolors.PowerNorm(0.5))
   cb = f.colorbar(cm, ax=ax)
   cb.set_label('count') 
   ax.set_yscale('log')
   ax.set_xlim(xlims[0], xlims[1])
   ax.set_ylim(ylims[0], ylims[1])  




   
genemax = int(sys.argv[1])
g9min = int(sys.argv[2])
g9max = int(sys.argv[3])
resolution = int(sys.argv[4])
threshold = int(sys.argv[5])
print('note that not all analytic calculations use the variables genemax, g9min and g9max -- these may return incorrect values if we deviate from genemax=3, g9min=1, g9max=8')
####################################################################################
string_for_saving_plot=str(genemax)+'_'+str(g9min)+'_'+str(g9max) +'_'+str(resolution) + '_'+str(threshold)
phenotypes=np.load('./biomorphs_map_pixels/phenotypes'+string_for_saving_plot+'.npy')
np.random.seed(4)
####################################################################################
print('get neutral set size, rho and phi_pq')
####################################################################################
no_of_different_vector_genes, no_of_different_g9s = 2*genemax+1, g9max-g9min+1
number_geno = np.product(phenotypes.shape)
g9_list=list(range(g9min, g9max+1))
####################################################################################
print('collect all GP map properties')
####################################################################################
filename_plotdata = './biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'.csv'
assert isfile('./biomorphs_map_pixels/arraycomplexitydata'+string_for_saving_plot+'.csv')
if not isfile('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_neutral_sets.csv'):
   neutral_set_size_dict = get_ph_vs_f(phenotypes)
   ph_list = [ph for ph in neutral_set_size_dict.keys()]
   N_list = [neutral_set_size_dict[ph] for ph in ph_list]
   array_complexity_df  = pd.read_csv('./biomorphs_map_pixels/arraycomplexitydata'+string_for_saving_plot+'.csv')
   ph_vs_array_complexity = {ph: complexity for ph, complexity in zip(array_complexity_df['phenotype integer'].tolist(), array_complexity_df['array complexity'].tolist())}
   ###
   filename_BDM = './biomorphs_map_pixels/block_decomp_arraycomplexitydata'+string_for_saving_plot+'_correlated.csv'
   array_complexity_bdm_df  = pd.read_csv(filename_BDM)
   ph_vs_array_complexity_bdm = {ph: complexity for ph, complexity in zip(array_complexity_bdm_df['phenotype integer'].tolist(), array_complexity_bdm_df['array complexity'].tolist())}
   ###
   df_global = pd.DataFrame.from_dict({'phenotype int': ph_list, 'neutral set size': N_list, 
                                        'array complexity': [ph_vs_array_complexity[ph] for ph in ph_list],
                                        'array complexity BDM': [ph_vs_array_complexity_bdm[ph] for ph in ph_list]})
   df_global.to_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_neutral_sets.csv')
   del N_list, df_global, ph_vs_array_complexity_bdm, ph_vs_array_complexity
else:
   print('loading', './biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_neutral_sets.csv')
   df_global = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_neutral_sets.csv')
   ph_list = [ph for ph in df_global['phenotype int'].tolist()]  
   neutral_set_size_dict  = df_data_to_dict('neutral set size', df_global)
   del df_global

if not isfile('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_local.csv'):
   Pevolvability = get_Pevolvability(phenotypes)
   max_evolv = max(Pevolvability.values())
   if g9min < 2 < g9max:
      g9_chosen_ph = 2
      g_chosen_ph = [g for g in np.random.randint(-1 * genemax, genemax, size=8)] + [g9_chosen_ph,]
      chosen_ph = phenotypes[tuple(g_chosen_ph)]
   else:
      potential_ph_for_phipq = [ph for ph in ph_list if 1000 <= Pevolvability[ph] < max_evolv - 1 and 5 * 10**4 < neutral_set_size_dict[ph]] #pick a phenotype with many connections to other phenotypes to observe range of phi pq, but do not pick maximum as this might be qualitatively different
      print(len(potential_ph_for_phipq), 'phenotypes meet the requirements for phipq')
      if len(potential_ph_for_phipq) > 0:
         chosen_ph = np.random.choice(potential_ph_for_phipq)
      else:
         chosen_ph = sorted(ph_list, key=Pevolvability.get, reverse=True)[3]
   rho, phi_pq = rho_phi_pq_biomorphs(phenotypes, chosen_ph)
   rho_list = [rho[ph] for ph in ph_list]
   phi_pq_list = [phi_pq[ph] for ph in ph_list]
   del rho, phi_pq
   Pevolvability_list  = [Pevolvability[ph] for ph in ph_list]
   del Pevolvability
   df_local = pd.DataFrame.from_dict({'phenotype int': ph_list, 'rho': rho_list, 
                               'phipq for phenotype '+str(chosen_ph): phi_pq_list,'Pevolvability': Pevolvability_list})
   df_local.to_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_local.csv')
   del  rho_list, phi_pq_list, Pevolvability_list, df_local
if not isfile(filename_plotdata):
   df_global = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_neutral_sets.csv')
   df_local = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_local.csv')
   for ph1, ph2 in zip(df_global['phenotype int'].tolist(), df_local['phenotype int'].tolist()):
      assert ph1 == ph2 #phenotypes in same order
   all_data = {column_name: df_local[column_name].tolist() for column_name in df_local.columns}
   for column_name in df_global.columns:
      all_data[column_name] = df_global[column_name].tolist()
   df = pd.DataFrame.from_dict(all_data)
   df.to_csv(filename_plotdata)

grob_filename = './biomorphs_map_pixels/genotype_robustness_array'+string_for_saving_plot+'.npy'
gevolv_filename = './biomorphs_map_pixels/genotype_evolvability_array'+string_for_saving_plot+'.npy'
if not isfile(gevolv_filename):
   Grobustness, Gevolvability = get_Grobustness_Gevolvability(phenotypes)
   np.save(grob_filename, Grobustness) 
   np.save(gevolv_filename, Gevolvability) 
else:
   Grobustness = np.load(grob_filename) 
   Gevolvability = np.load(gevolv_filename)
####################################################################################
print('load')
####################################################################################
df = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'.csv')
neutral_set_size_dict  = df_data_to_dict('neutral set size', df)
rho  = df_data_to_dict('rho', df)
phi_pq_column_label = [c for c in df.columns if c.startswith('phipq')][0]
chosen_ph = int(phi_pq_column_label.split()[-1])
phi_pq  = df_data_to_dict(phi_pq_column_label, df)
Pevolvability  = df_data_to_dict('Pevolvability', df)

print('number of phenotypes with N > 10', len([N for N in neutral_set_size_dict.keys() if neutral_set_size_dict[N] > 10]))
print('total number of phenotypes', len([N for N in neutral_set_size_dict.keys()]))
print('total number of phenotypes analytic model', sum([7**8/(2 * 7**(9 - 2 * g9)) if g9 <= 4 else 7**8/2 for g9 in range(1, 9)]))
print('highest rank in analytic model', 2 * 7**8)
print('number of phenotypes with N = 2', len([N for N in neutral_set_size_dict.keys() if neutral_set_size_dict[N] == 2]))
print('minimum phenotype evolvability', min([e for e in Pevolvability.values()]))
print('median phenotype evolvability', np.median([e for e in Pevolvability.values()]))
print('neutral set size of beetle', neutral_set_size_dict[2420233])
####################################################################################
print('plot phenotype for phi pq', flush=True)
####################################################################################
number_pheno = len(neutral_set_size_dict.values())
print('evolvability of p ', Pevolvability[chosen_ph], '- normalised fraction inaccessible', 1- Pevolvability[chosen_ph]/float(number_pheno - 1))
print('number with higher evolvability', len([p for p in Pevolvability if Pevolvability[p] > Pevolvability[chosen_ph]]), [Pevolvability[p] for p in Pevolvability if Pevolvability[p] > Pevolvability[chosen_ph]])
g9_chosen_ph = min([g[-1] for g, ph in np.ndenumerate(phenotypes) if ph == chosen_ph]) + g9min
print('neutral set size of p', neutral_set_size_dict[chosen_ph], 'analytic model', [2 if g9 >= 5 else 2* no_of_different_vector_genes**(9-2*g9) for g9 in [g9_chosen_ph]][0])
print('g9_chosen_ph',g9_chosen_ph )
f, ax=plt.subplots()
plot_phenotype(chosen_ph, ax, phenotypes, g9min, color='b', linewidth=7)
f.savefig('./plots_pixel_def/chosen_ph_phipq_'+str(chosen_ph)+'_phenotypes_'+string_for_saving_plot+'.eps', bbox_inches='tight')
plt.close()
del f, ax
####################################################################################
print('plot a few biomprhs_as_subplots')
####################################################################################
for ncols, nrows in [(3, 4), (4, 3), (10, 3)]:
   phenotypes_sorted_by_NSS= sorted([N for N in neutral_set_size_dict.keys()], key=neutral_set_size_dict.get, reverse=True)
   for i in range(3):
      f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 2.4, nrows * 2.6))
      if i == 0:
         ph_list = [phenotypes_sorted_by_NSS[int(10**i)] for i in np.linspace(np.log10(3), np.log10(len(phenotypes_sorted_by_NSS) - 1), ncols*nrows)]
      elif i == 1:
         number_rare = ncols
         phenotypes_sorted_by_NSS_except_rarest = sorted([N for N in neutral_set_size_dict.keys() if neutral_set_size_dict[N] > 2.5], key=neutral_set_size_dict.get, reverse=True)
         phenotypes_list_rarest = sorted([N for N in neutral_set_size_dict.keys() if neutral_set_size_dict[N] < 2.5], key=neutral_set_size_dict.get, reverse=True)
         ph_list_unsorted = [phenotypes_sorted_by_NSS_except_rarest[int(10**i)] for i in np.linspace(np.log10(3), np.log10(len(phenotypes_sorted_by_NSS_except_rarest) -1), ncols*nrows - number_rare)] # end does not matter so much since there are millions of phenotypes with lowest neutral set size value
         ph_list_unsorted = ph_list_unsorted + [phenotypes_list_rarest[int(i)] for i in np.linspace(0, len(phenotypes_list_rarest) - 1, number_rare)] # there will be one very rare one in cross-section, so choose one less
         ph_list = sorted(ph_list_unsorted, key=neutral_set_size_dict.get, reverse=True)
      elif i == 2:
         ph_list = phenotypes_sorted_by_NSS[:ncols*nrows]
         f.suptitle('plots of most frequent phenotypes')
      assert len(ph_list) == len(set(ph_list))
      for phindex, ph in enumerate(ph_list):
         plotindex = (phindex//ncols, phindex%ncols)
         plot_phenotype(ph, ax[plotindex], phenotypes, g9min, color='k')
         ax[plotindex].set_title(decimal_to_scientific_notation(neutral_set_size_dict[ph], 0), fontsize=17)# + ' genotypes')
      #f.tight_layout()
      f.subplots_adjust(wspace=0.4, hspace=0.5)
      f.savefig('./plots_pixel_def/biomorph_examples'+string_for_saving_plot+ '_' + str(i) + '_' + '_'.join([str(ncols), str(nrows)]) + '.eps', bbox_inches='tight')
      plt.close()
      del f, ax
####################################################################################
print('plot a few biomorphs as example - including high/low complexity')
####################################################################################
complexity_N_category_vs_ph = {(c, N): [] for c in ['high', 'low', 'medium'] for N in ['high', 'low', 'medium']}
half_max_log_N = 0.5 * (np.log10(max([N for N in neutral_set_size_dict.values()])) + np.log10(min([N for N in neutral_set_size_dict.values()])))
ph_vs_arrayKC  = df_data_to_dict('array complexity BDM', df)

half_range_KC = 0.5 * (min([N for N in ph_vs_arrayKC.values()]) + max([N for N in ph_vs_arrayKC.values()]))
f, ax = plt.subplots(ncols=4, nrows=3, figsize=(4 * 2.4, 3 * 2.4))
for ph in neutral_set_size_dict.keys():
   if ph_vs_arrayKC[ph] > 1.5 * half_range_KC:
       K = 'high'
   elif ph_vs_arrayKC[ph] < 0.5 * half_range_KC:
       K= 'low'
   else:
       K = 'medium'
   if np.log10(neutral_set_size_dict[ph]) > 1.5 * half_max_log_N:
       N = 'high'
   elif np.log10(neutral_set_size_dict[ph]) < 0.5 * half_max_log_N:
       N = 'low'
   else:
      N = 'medium'
   complexity_N_category_vs_ph[(K, N)].append(ph)

for rowindex, category in enumerate([('low', 'low'), ('low', 'high'), ('high', 'low')]):
   phenotypes_to_plot = np.random.choice(complexity_N_category_vs_ph[category], min(4, len(complexity_N_category_vs_ph[category])), replace=False)
   for i, ph in enumerate(phenotypes_to_plot):
      ph = phenotypes_to_plot[i]
      if ph == phenotypes[tuple([genemax,] * 8 + [g9min,])]:
         ax[rowindex, i].scatter(0, 0, c='k', s=30)
         ax[rowindex, i].set_aspect('equal', 'box')
         ax[rowindex, i].axis('off')
      else:
         plot_phenotype(ph, ax[rowindex, i], phenotypes, g9min, color='k') 
      ax[rowindex, i].set_title(decimal_to_scientific_notation(neutral_set_size_dict[ph], 0) + ' genotypes\n'+ 'complexity = '+ decimal_to_scientific_notation(ph_vs_arrayKC[ph], 0))
   for i in range(len(phenotypes_to_plot), 4):
      ax[rowindex, i].axis('off')
f.tight_layout()
f.savefig('./plots_pixel_def/lowKlowP_examples'+string_for_saving_plot+ '_' + str(i) + '_' + '_'.join([str(ncols), str(nrows)]) + '.eps', bbox_inches='tight')
plt.close()
del f, ax

####################################################################################
print('data as lists')
####################################################################################
phenotype_list_data = [N for N in neutral_set_size_dict.keys()]
rho_list_data = [rho[pheno] for pheno in phenotype_list_data]
neutral_set_size_list_data = [neutral_set_size_dict[pheno] for pheno in phenotype_list_data]
frequency_list_data=np.divide(neutral_set_size_list_data, float(number_geno))
Pevolvability_list_data=[Pevolvability[pheno] for pheno in phenotype_list_data]

phi_pq_list_data=[phi_pq[pheno] if pheno in phi_pq else 0 for pheno in phenotype_list_data ]

####################################################################################
print('all bias plots')
####################################################################################
for array_complexity in ['array complexity', 'array complexity BDM']:
   ph_vs_arrayKC  = df_data_to_dict(array_complexity, df)
   complexity_list_data = [ph_vs_arrayKC[pheno] for pheno in phenotype_list_data]
    ###
   label = 'neutral set size'
   f, (ax0, ax1) =plt.subplots(nrows=2, figsize=(5,8))
   ## neutral set size vs rank
   ax0.set_ylabel(label, fontsize=20)
   ax0.set_xlabel('rank', fontsize=20)
   sorted_freq_data = np.array(sorted(neutral_set_size_list_data, reverse=True), dtype='int')
   N_vs_rank_data = {f: (sorted_freq_data >= f - 0.01).sum() for f in set(sorted_freq_data)}
   rank_data = [N_vs_rank_data[f] for f in sorted_freq_data]
   discrete_N_model = [2 * no_of_different_vector_genes**(9 - 2*g9) for g9 in range(1, 5)] + [2]
   ax0.plot([7**8/N if N > 3 else 2*no_of_different_vector_genes**8 for N in discrete_N_model], discrete_N_model, c='r', marker='x', ms=10)
   ax0.scatter(rank_data, sorted_freq_data, c='b', s=3)
   ax0.set_xscale('log')
   ax0.set_yscale('log')
   ymin = 1
   ax0.set_ylim(ymin, max(neutral_set_size_list_data) * 1.5)
   ax0.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
   ax0.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
   ax0.xaxis.set_minor_locator(ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 25))
   ax0.xaxis.set_minor_formatter(ticker.NullFormatter())
   ax0.yaxis.set_minor_locator(ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 25))
   ax0.yaxis.set_minor_formatter(ticker.NullFormatter())
   
   ## complexity versus frequency
   ax1.set_ylabel(label, fontsize=20)
   ax1.set_xlabel('complexity estimate', fontsize=20)
   linlog2Dhist(complexity_list_data, neutral_set_size_list_data, ax1, f)
   ax1.set_yscale('log')
   slope, intercept = approximate_upper_bound_complexity_plot(complexity_list_data, neutral_set_size_list_data)
   print(slope, intercept)
   if slope:
      #if not normalised:
      x_list = np.linspace(min(complexity_list_data), (1 - intercept)/slope)
      ax1.plot(x_list, np.power(2, np.array(x_list)*slope + intercept), c='k')
   ax1.set_ylim(ymin, max(neutral_set_size_list_data) * 1.5)
   ax1.set_xlim(0.75*min(complexity_list_data), 1.25*max(complexity_list_data))
   ax1.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
   ax1.yaxis.set_minor_locator(ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 25))
   ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
   ## format
   [axi.tick_params(axis='both', which='major', labelsize=16) for axi in [ax0, ax1]]
   #[axi.tick_params(axis='both', which='minor', labelsize=18) for axi in [ax0, ax1]]
   f.tight_layout()
   f.savefig('./plots_pixel_def/bias'+string_for_saving_plot+'_'.join(array_complexity.split()) +'.png', dpi=300, bbox_inches='tight')
   plt.close('all')
   del f, ax0, ax1

# for inset - complexity in model
complexity_list_discrete_N = [np.ceil(np.log2(no_of_different_vector_genes)) * (9 - np.log(N/2.0)/np.log(no_of_different_vector_genes)) for N in discrete_N_model]

f, ax = plt.subplots(figsize=(4, 3))
ax.set_ylabel('neutral set size', fontsize=20)
ax.set_xlabel('complexity estimate', fontsize=20)
ax.scatter(complexity_list_discrete_N, discrete_N_model, edgecolor='none', s=80, c='r', zorder=2)
ax.set_yscale('log')
ax.plot(complexity_list_discrete_N, discrete_N_model, c='k', zorder=1)
ax.set_ylim(1, max(discrete_N_model) * 1.5)
ax.set_xlim(0.75*min(complexity_list_discrete_N), 1.25*max(complexity_list_discrete_N))
f.tight_layout()
f.savefig('./plots_pixel_def/smplicity_bias_model'+string_for_saving_plot+'.png', dpi=300, bbox_inches='tight')
plt.close('all')
del f, ax
# for comparison - number of lines in model
number_lines_list_discrete_N = [2**g9 -1 for g9 in range(g9min, g9max)]
discrete_N_model_for_number_lines = [2 * no_of_different_vector_genes**(9 - 2 * g9) if g9 <= 4 else 2 for g9 in range(g9min, g9max)]
f, ax = plt.subplots(figsize=(4, 3))
ax.set_ylabel('neutral set size', fontsize=16)
ax.set_xlabel('number of lines', fontsize=16)
ax.scatter(number_lines_list_discrete_N, discrete_N_model_for_number_lines, edgecolor='none', s=80, c='r', zorder=2)
ax.set_yscale('log')
ax.plot(number_lines_list_discrete_N, discrete_N_model_for_number_lines, c='k', zorder=1)
ax.set_ylim(1, max(discrete_N_model_for_number_lines) * 1.5)
ax.set_xlim(0.75*min(number_lines_list_discrete_N), 1.25*max(number_lines_list_discrete_N))
f.tight_layout()
f.savefig('./plots_pixel_def/number_lines_model'+string_for_saving_plot+'.png', dpi=300, bbox_inches='tight')
plt.close('all')
del f, ax

####################################################################################
print('P(K) plot')
####################################################################################
for array_complexity in ['array complexity', 'array complexity BDM']:
   ph_vs_arrayKC  = df_data_to_dict(array_complexity, df)
   list_all_complexities = [ph_vs_arrayKC[pheno] for pheno in phenotype_list_data for N in range(neutral_set_size_dict[pheno])]
   f, ax = plt.subplots(figsize=(3.5, 2.5))
   ax.hist(list_all_complexities, density=False, weights= [1/float(len(list_all_complexities)) for c in list_all_complexities], edgecolor='black', color='grey')
   ax.set_xlabel('estimated complexity')
   ax.set_ylabel('fraction of genotypes')
   ax.set_yscale('log')
   #ax.set_ylim(0.5, len() * )
   f.tight_layout()
   f.savefig('./plots_pixel_def/K_p_plot_log'+string_for_saving_plot+'_'.join(array_complexity.split())+'.png', dpi=300, bbox_inches='tight')
   plt.close('all')
   del f, ax
####################################################################################
print('P_p(K) plot (distribution over phenotypes)')
####################################################################################
for array_complexity in ['array complexity', 'array complexity BDM']:
   ph_vs_arrayKC  = df_data_to_dict(array_complexity, df)
   list_all_complexities = [ph_vs_arrayKC[pheno] for pheno in phenotype_list_data]
   f, ax = plt.subplots(figsize=(3.5, 2.5))
   ax.hist(list_all_complexities, density=False, weights= [1/float(len(list_all_complexities)) for c in list_all_complexities], edgecolor='black', color='grey')
   ax.set_xlabel('estimated complexity')
   ax.set_ylabel('fraction of phenotypes')
   ax.set_yscale('log')
   #ax.set_ylim(0.5, len() * )
   f.tight_layout()
   f.savefig('./plots_pixel_def/phenotypic_K_p_plot_log'+string_for_saving_plot+'_'.join(array_complexity.split())+'.png', dpi=300, bbox_inches='tight')
   plt.close('all')
   del f, ax

####################################################################################
print('all mutation plots')
####################################################################################
f = plt.figure(figsize=(14,8))
gs = GridSpec(ncols=6, nrows=2)
ax0 = f.add_subplot(gs[0, :3])
ax1 = f.add_subplot(gs[0, 3:])
ax3 = f.add_subplot(gs[1, :2])
ax4 = f.add_subplot(gs[1, 2:4])
ax2 = f.add_subplot(gs[1, 4:])
ax0.axis('off')
## robustness and frequency
ax1.set_ylabel(r'phenotype robustness, $\rho_p$', fontsize=18)
ax1.set_xlabel(r'phenotype frequency, $f_{p}$', fontsize=18)
ax1.set_xscale('log')
ax1.plot(np.linspace(0.2*min(frequency_list_data), 1.0), np.linspace(0.2*min(frequency_list_data), 1.0), c='k')
discrete_N_model = [2 * no_of_different_vector_genes**(9 - 2*g9) for g9 in range(1, 5)] + [2]
discrete_rhop_model = [1/9 * (np.log(N/2)/np.log(no_of_different_vector_genes)) for N in discrete_N_model]
print('discrete_rhop_model', discrete_rhop_model)
ax1.plot(np.divide(discrete_N_model, no_of_different_vector_genes**8 * no_of_different_g9s), discrete_rhop_model, c='r', marker='x', ms=10)
ax1.scatter(frequency_list_data, rho_list_data, s=15, c='b', lw=0)
ax1.set_xlim(0.5*min(frequency_list_data), 1.05)
## phi_pq and frequency
ax2.set_ylabel(r' mutation prob., $\phi_{pq}$', fontsize=18)
ax2.set_xlabel(r'phenotype frequency, $f_{p}$', fontsize=18)
ax2.set_xscale('log')
ax2.set_yscale('log')
x_axis_linspace_phipq = np.linspace(0.2*min(frequency_list_data), 1.0)
ax2.plot(x_axis_linspace_phipq, x_axis_linspace_phipq, c='k')
#ax2.plot(freq_list_phipq_model, phipq_model, c='r')
ax2.plot([2 * 7**(9-2*g9)/(7**8 * 8) if g9 <= 4 else 2/(7**8 * 8) for g9 in [g9_chosen_ph -1, g9_chosen_ph, g9_chosen_ph +1]], 
         [1/18,] * 2 + [1/(18* 7**2) if g9 <=3 else 1/(18* 7) if g9 == 4 else 1/18 for g9 in [g9_chosen_ph,]], 
         c='r', marker='x', ms=10)
nonzero_frequency_list_data, nonzero_phi_pq_list_data = zip(*[(f, phi) for f, phi in zip(frequency_list_data, phi_pq_list_data) if not np.isnan(phi) and phi > 0.5/(20 * number_geno)])
print('number of non-zero phipq', len(nonzero_phi_pq_list_data), 'pevolv', Pevolvability[chosen_ph] , flush=True)
ax2.set_ylim(min(nonzero_phi_pq_list_data + nonzero_frequency_list_data) * 0.5, 2 * max(nonzero_phi_pq_list_data + nonzero_frequency_list_data))
ax2.set_xlim(min(nonzero_phi_pq_list_data + nonzero_frequency_list_data) * 0.5, 2 * max(nonzero_phi_pq_list_data + nonzero_frequency_list_data))
ax2.scatter(frequency_list_data, phi_pq_list_data, s=15, c='b')
## geno eolvability/robustness
ax3.set_xlabel(r'genotype robustness, $\widetilde{\rho}_g$', fontsize=18)
ax3.set_ylabel(r'genotype evolvability, $\widetilde{\epsilon}_g$', fontsize=18)
linlin2Dhist(np.ravel(Grobustness), np.ravel(Gevolvability), ax3, f, norm='log')
ax3.set_ylim(-0.5, max(np.ravel(Gevolvability)) * 1.1)
ax3.plot(discrete_rhop_model, [18 * (1-rho) for rho in discrete_rhop_model], c='r', marker='x', ms=10)
#ax3.plot([g9_vs_rob_model[g9] for g9 in g9_list], [g9_vs_gev_model[g9] for g9 in g9_list], c='r')
## pheno eolvability/robustness
ax4.set_xlabel(r'phenotype robustness, $\rho_p$', fontsize=18)
ax4.set_ylabel(r'phenotype evolvability, $\epsilon_p$', fontsize=18)
ax4.set_yscale('log')
linlog2Dhist(rho_list_data, Pevolvability_list_data, ax4, f)
print('strength of pheno robustness/log evolvability correlation (computational)', pearsonr(rho_list_data, np.log10(Pevolvability_list_data)))
Pevolvability_list_model = [18 if rho < 0.01 else 15 + 7 if rho < 0.01 + 1/9 else 18 * (1 - rho) - 1 + 7**2 for rho in discrete_rhop_model]
print('strength of pheno robustness/log evolvability correlation (analytic)', pearsonr(discrete_rhop_model, np.log10(Pevolvability_list_model)))
ax4.plot(discrete_rhop_model, Pevolvability_list_model, c='r', marker='x', ms=10)
#ax4.plot([g9_vs_rob_model[g9] for g9 in g9_list], [g9_vs_pevolv_model[g9] for g9 in g9_list], c='r')

## format
[axi.tick_params(axis='both', which='major', labelsize=18) for axi in [ax1, ax2, ax3, ax4]]
[axi.tick_params(axis='both', which='minor', labelsize=18) for axi in [ax1, ax2, ax3, ax4]]
[axi.annotate('ABCDEFGH'[i + 1], xy=(0.05, 0.88), xycoords='axes fraction', fontsize=25, fontweight='bold') for i, axi in enumerate([ax1, ax3, ax4, ax2])]
f.tight_layout()
f.savefig('./plots_pixel_def/mutations'+string_for_saving_plot+'.png', dpi=300, bbox_inches='tight')
plt.close('all')
del ax1, ax2, ax3, ax4, f, ax0
"""
####################################################################################
print('all plots')
####################################################################################
f, ax = plt.subplots(ncols=3, nrows=2, figsize=(14,8))
# neutral set size vs rank
ax[0, 0].set_ylabel('neutral set size', fontsize=20)
ax[0, 0].set_xlabel('rank', fontsize=20)
#ax[0, 0].plot(np.arange(1, 1+len(neutral_set_size_list_model)), sorted(neutral_set_size_list_model, reverse=True),c='r')
#discrete_N_model = [2 * 7**(9 - 2*g9) for g9 in range(1, 5)] + [2]
ax[0, 0].plot([7**8/N if N > 3 else 4*7**8/N for N in discrete_N_model], discrete_N_model,c='r', marker='x', ms=10)
sorted_freq_data = np.array(sorted(neutral_set_size_list_data, reverse=True), dtype='int')
N_vs_rank_data = {f: (sorted_freq_data >= f - 0.01).sum() for f in set(sorted_freq_data)}
rank_data = [N_vs_rank_data[f] for f in sorted_freq_data]
ax[0, 0].scatter(rank_data, sorted_freq_data, c='b', s=3)
ax[0, 0].scatter(np.arange(1, 1+len(neutral_set_size_list_data)), sorted(neutral_set_size_list_data, reverse=True), c='b', s=3)
ax[0, 0].set_xscale('log')
ax[0, 0].set_yscale('log')
ax[0, 0].set_ylim(1, max(neutral_set_size_list_data) * 1.5)
## complexity versus frequency
ax[0, 1].set_ylabel('neutral set size', fontsize=20)
ax[0, 1].set_xlabel('complexity estimate', fontsize=20)
linlog2Dhist(complexity_list_data, neutral_set_size_list_data, ax[0, 1], f)
ax[0, 1].set_yscale('log')
slope, intercept = approximate_upper_bound_complexity_plot(complexity_list_data, neutral_set_size_list_data)
if slope:
   x_list = np.linspace(min(complexity_list_data), (1 - intercept)/slope)
   ax[0, 1].plot(x_list, np.power(2, np.array(x_list)*slope + intercept), c='k')
ax[0, 1].set_ylim(1, max(neutral_set_size_list_data) * 1.5)
ax[0, 1].set_xlim(0.75*min(complexity_list_data), 1.25*max(complexity_list_data))

## robustness and frequency
ax[0, 2].set_ylabel('phenotype robustness', fontsize=20)
ax[0, 2].set_xlabel('phenotype frequency', fontsize=20)
ax[0, 2].set_xscale('log')
ax[0, 2].plot(np.linspace(0.2*min(frequency_list_data), 1.0), np.linspace(0.2*min(frequency_list_data), 1.0), c='k')
#ax[0, 2].plot([g9_vs_freq_model[g9] for g9 in g9_list], [g9_vs_rob_model[g9] for g9 in g9_list], c='r')
ax[0, 2].plot(np.divide(discrete_N_model, 7**8 * 8.0), discrete_rhop_model, c='r', marker='x', ms=10)
ax[0, 2].scatter(frequency_list_data, rho_list_data, s=3, c='b', lw=0)
ax[0, 2].set_xlim(0.5*min(frequency_list_data), 1.05)
## phi_pq and frequency
ax[1, 0].set_ylabel(r' mutation prob. $\phi_{pq}$', fontsize=20)
ax[1, 0].set_xlabel(r'phenotype frequency, $f_{p}$', fontsize=20)
ax[1, 0].set_xscale('log')
ax[1, 0].set_yscale('log')
x_axis_linspace_phipq = np.linspace(0.2*min(frequency_list_data), 1.0)
ax[1, 0].plot(x_axis_linspace_phipq, x_axis_linspace_phipq, c='k')
#ax[1, 0].plot(freq_list_phipq_model, phipq_model, c='r')
nonzero_frequency_list_data, nonzero_phi_pq_list_data = zip(*[(freq, phi) for freq, phi in zip(frequency_list_data, phi_pq_list_data) if not np.isnan(phi) and phi > 0.5/(20 * number_geno)])
ax[1, 0].set_ylim(min(nonzero_phi_pq_list_data + nonzero_frequency_list_data) * 0.5, 2 * max(nonzero_phi_pq_list_data + nonzero_frequency_list_data))
ax[1, 0].set_xlim(min(nonzero_phi_pq_list_data + nonzero_frequency_list_data) * 0.5, 2 * max(nonzero_phi_pq_list_data + nonzero_frequency_list_data))
ax[1, 0].scatter(frequency_list_data, phi_pq_list_data, s=3, c='b')
ax[1, 0].plot([2 * 7**(9-2*g9)/(7**8 * 8) if g9 <= 4 else 2/(7**8 * 8) for g9 in [g9_chosen_ph -1, g9_chosen_ph, g9_chosen_ph +1]], 
         [1/18,] * 2 + [1/(18* 7**2) if g9 <=3 else 1/(18* 7) if g9 == 4 else 1/18 for g9 in [g9_chosen_ph,]], 
         c='r', marker='x', ms=10)
assert len(nonzero_phi_pq_list_data) == Pevolvability[chosen_ph]
high_freq_phipq_point = [(pheno, freq, phi) for pheno, freq, phi in zip(phenotype_list_data, frequency_list_data, phi_pq_list_data) if not np.isnan(phi) and phi > 0.5/(20 * number_geno) and freq > 0.01]
for pheno, freq, phi in high_freq_phipq_point:
    f2, ax2=plt.subplots()
    plot_phenotype(pheno, ax2, phenotypes, g9min, color='b', linewidth=7)
    ax2.set_title(r'$f_q = $'+str(freq) + r'  $\phi_{pq} = $'+str(phi))
    f2.savefig('./plots_pixel_def/high_f_in_phipq_plot'+str(chosen_ph)+'_'+str(pheno)+'_phenotypes_'+string_for_saving_plot+'.eps', bbox_inches='tight')
    del f2, ax2
## geno eolvability/robustness
ax[1, 1].set_xlabel('genotype robustness', fontsize=20)
ax[1, 1].set_ylabel('genotype evolvability', fontsize=20)
linlin2Dhist(np.ravel(Grobustness), np.ravel(Gevolvability), ax[1, 1], f, norm='log')
ax[1, 1].set_ylim(-0.5, max(np.ravel(Gevolvability)) * 1.1)
ax[1, 1].plot(discrete_rhop_model, [18 * (1-rho) for rho in discrete_rhop_model], c='r', marker='x', ms=10)
## pheno eolvability/robustness
ax[1, 2].set_xlabel('phenotype robustness', fontsize=20)
ax[1, 2].set_ylabel('phenotype evolvability', fontsize=20)
ax[1, 2].set_yscale('log')
linlog2Dhist(rho_list_data, Pevolvability_list_data, ax[1, 2], f)
ax[1, 2].plot(discrete_rhop_model, [18 if rho < 0.01 else 15 + 7 if rho < 0.01 + 1/9 else 18 * (1 - rho) - 1 + 7**2 for rho in discrete_rhop_model], c='r', marker='x', ms=10)

## format
[axi.tick_params(axis='both', which='major', labelsize=18) for i, axi in np.ndenumerate(ax)]
[axi.tick_params(axis='both', which='minor', labelsize=18) for i, axi in np.ndenumerate(ax)]
[axi.annotate('ABCDEFGH'[np.sum(i)], xy=(0.05, 0.88), xycoords='axes fraction', fontsize=25, fontweight='bold') for i, axi in np.ndenumerate(ax)]
f.tight_layout()
f.savefig('./plots_pixel_def/all_plots_with_model'+string_for_saving_plot+'.png', dpi=300, bbox_inches='tight')
plt.close('all')
del f, ax

################################################################################################################################################################################################
print('shape space covering property - no periodic boundary conditions')
################################################################################################################################################################################################
number_samples = 10
filename_shapespace_covering = './biomorphs_map_pixels/shape_space_covering'+string_for_saving_plot+'_'+str(number_samples)+'.csv'
filename_shapespace_covering_geno = './biomorphs_map_pixels/shape_space_covering'+string_for_saving_plot+'_'+str(number_samples)+'_geno.csv'
#####################################
if not isfile(filename_shapespace_covering):
   number_pheno = len(neutral_set_size_dict.values())
   frequency_threshold_common_phenotype = number_geno/float(number_pheno)
   common_phenotypes_list = [pheno for pheno, N in neutral_set_size_dict.items() if N >= frequency_threshold_common_phenotype]
   max_geno_dist = np.sum(phenotypes.shape) - len(phenotypes.shape)
   distance_vs_fraction_common_phenotypes = {d: [] for d in range(1, max_geno_dist + 1)}
   distance_vs_number_genotypes = {d: [] for d in range(1, max_geno_dist + 1)}
   for repetition in range(number_samples):
      distance_vs_number_genotypes_thisrun = {d: 0 for d in range(0, max_geno_dist + 1)}
      print('repeat shape space covering analysis for the', repetition, 'th time')
      genotype_start = random_startgene(genemax, g9min, g9max)
      distance_vs_list_common_phenotypes = {d: set([]) for d in range(max_geno_dist + 1)}
      for genotype, ph in np.ndenumerate(phenotypes):
         d = sum([abs(i - j) for i, j in zip(genotype_start, genotype)])
         distance_vs_number_genotypes_thisrun[d] += 1
         if neutral_set_size_dict[ph] >= frequency_threshold_common_phenotype:
            distance_vs_list_common_phenotypes[d].add(ph)
      for d in distance_vs_fraction_common_phenotypes:
         set_all_phenot_up_to_d = set([ph for d_less in range(1, d + 1) for ph in distance_vs_list_common_phenotypes[d_less]])
         distance_vs_fraction_common_phenotypes[d].append(len(set_all_phenot_up_to_d)/float(len(common_phenotypes_list)))
         distance_vs_number_genotypes[d].append(sum([distance_vs_number_genotypes_thisrun[d_less] for d_less in range(0, d + 1)])/float(number_geno))
         del set_all_phenot_up_to_d
   print({'d=' + str(k): len(l[:]) for k, l in distance_vs_fraction_common_phenotypes.items()})
   print({'d=' + str(k): len(l[:]) for k, l in distance_vs_number_genotypes.items()})
   df_pheno_shape_space_covering = pd.DataFrame.from_dict({'d=' + str(k): l[:] for k, l in distance_vs_fraction_common_phenotypes.items()})
   df_geno_shape_space_covering = pd.DataFrame.from_dict({'d=' + str(k): l[:] for k, l in distance_vs_number_genotypes.items()})
   df_pheno_shape_space_covering.to_csv(filename_shapespace_covering)
   df_geno_shape_space_covering.to_csv(filename_shapespace_covering_geno)
else:
   df_pheno_shape_space_covering = pd.read_csv(filename_shapespace_covering)
   distance_vs_fraction_common_phenotypes = {int(c[2:]): df_pheno_shape_space_covering[c].tolist()[:] for c in df_pheno_shape_space_covering.columns if c.startswith('d=')}
   df_geno_shape_space_covering = pd.read_csv(filename_shapespace_covering_geno)   
   distance_vs_number_genotypes = {int(c[2:]): df_geno_shape_space_covering[c].tolist()[:] for c in df_geno_shape_space_covering.columns if c.startswith('d=')}
   max_geno_dist = max(distance_vs_number_genotypes.keys())


##################
## also include fraction of all genotypes at this distance
################
f, ax = plt.subplots(figsize=(4.5, 3))
ax.errorbar([d for d in range(1, max_geno_dist)],
            [np.mean(distance_vs_fraction_common_phenotypes[d]) for d in range(1, max_geno_dist)],
            yerr=[np.std(distance_vs_fraction_common_phenotypes[d]) for d in range(1, max_geno_dist)], label='fraction of common phenotypes')
ax.errorbar([d for d in range(1, max_geno_dist)],
            [np.mean(distance_vs_number_genotypes[d]) for d in range(1, max_geno_dist)],
            yerr=[np.std(distance_vs_number_genotypes[d]) for d in range(1, max_geno_dist)], label='fraction of all genotypes (for reference)')
ax.set_xlabel(r'mutational distance $d$')
ax.set_ylabel('fraction that is contained\n within distance' + r' $d$')
f.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18))
f.tight_layout()
f.savefig('./plots_pixel_def/shape_space_covering'+string_for_saving_plot+'number_samples'+str(number_samples)+'_with_fraction_genotypes.png', bbox_inches='tight')
plt.close('all')
del f, ax


####################################################################################
print('simplifity bias and phipq')
####################################################################################
Pevolvability  = df_data_to_dict('Pevolvability', df)
if not isfile('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_phipq.csv'):
   phenotype_list_data_for_phipq = [ph for ph in ph_list if 100 <= Pevolvability[ph] ]
   print('number with sufficiently high P evolv:', len(phenotype_list_data_for_phipq))
   ph_vs_phipq = phi_pq_list_initial_ph(phenotypes, neutral_set_size_dict, phenotype_list_data_for_phipq)
   pheno_one_list, pheno_two_list, phi_pq_list = zip(*[(phenoone, phenotwo, phi) for phenoone in phenotype_list_data_for_phipq for phenotwo, phi in ph_vs_phipq[phenoone].items()])
   df_phi_pq = pd.DataFrame.from_dict({'phenotype 1': pheno_one_list, 'phenotype 2': pheno_two_list, 'phipq': phi_pq_list})
   df_phi_pq.to_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_phipq.csv')
   del  pheno_one_list, pheno_two_list, phi_pq_list
else:
   df_phi_pq = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'_phipq.csv')
   phenotype_list_data_for_phipq = list(set([p for p in df_phi_pq['phenotype 1'].tolist()]))
   ph_vs_phipq = {ph: {} for ph in phenotype_list_data_for_phipq}
   for rowindex, row in df_phi_pq.iterrows():
      ph_vs_phipq[row['phenotype 1']][row['phenotype 2']] = row['phipq']
####################################################################################
df_filename = './biomorphs_map_pixels/'+string_for_saving_plot+'_phipq_complexity.csv'
if not isfile(df_filename):
   high_phi_complexity, complexity_all_nonzero = [], []
   for firstpheno in phenotype_list_data_for_phipq:
      phenotype_list_two = [secondpheno for secondpheno in ph_vs_phipq[firstpheno].keys()]
      phhenotype_two_by_phi = sorted(phenotype_list_two[:], key = ph_vs_phipq[firstpheno].get, reverse=True)
      complexity_list_data_by_phi = [ph_vs_arrayKC[secondpheno] for secondpheno in phhenotype_two_by_phi]
      high_phi_complexity.append(np.mean(complexity_list_data_by_phi[:10]))
      complexity_all_nonzero.append(np.mean(complexity_list_data_by_phi))
   df_phipq_complexity_corr = pd.DataFrame.from_dict({
                              'phenotype p': phenotype_list_data_for_phipq,
                              'complexity of high-phi phenotypes': high_phi_complexity,
                              'complexity of average phenotypes with non-zero phi': complexity_all_nonzero})
   df_phipq_complexity_corr.to_csv(df_filename)
else:
   df_phipq_complexity_corr = pd.read_csv(df_filename)
f, ax=plt.subplots(figsize=(4,4))
ax.set_ylabel('mean complexity of\n'+r'phenotypyes $p$' +'\n'+r'with $10$ highest $\phi_{pq}$')#, fontsize=20)
ax.set_xlabel('mean complexity of\n'+r'all phenotypyes $p$' +'\n'+r'with non-zero $\phi_{pq}$')#, fontsize=20)
ax.set_title(r'one value per initial phenotype $q$'+'\n'+ r'for phenotypes $q$ with at least'+'\n100 distince phenotypic transitions')#, fontsize=20)
complexity_all_nonzero, high_phi_complexity = df_phipq_complexity_corr['complexity of average phenotypes with non-zero phi'].tolist(), df_phipq_complexity_corr['complexity of high-phi phenotypes'].tolist()
ax.scatter(complexity_all_nonzero, high_phi_complexity, s=10, edgecolor='none', c='b')
min_c, max_c = 0.8 * min(complexity_all_nonzero + high_phi_complexity), 1.2 * max(complexity_all_nonzero + high_phi_complexity)
ax.plot([min_c, max_c], [min_c, max_c], c='k') 
ax.set_xlim(min_c, max_c)
ax.set_ylim(min_c, max_c)    
f.tight_layout()
f.savefig('./plots_pixel_def/mutations_complexity_change'+string_for_saving_plot+'.png', dpi=300, bbox_inches='tight')
plt.close('all')
del f, ax"""




