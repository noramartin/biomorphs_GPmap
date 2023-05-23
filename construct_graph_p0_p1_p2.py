#!/usr/bin/env python3
import matplotlib
matplotlib.use('agg')
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
from functions_for_GP_analysis.GPproperties import plot_phenotype, neighbours_g_indices_nonperiodic_boundary, df_data_to_dict
from functions_for_GP_analysis.neutral_component import get_NC_genotypes, NC_rho_c_pq_biomorphs
from os.path import isfile
import definition_special_genotypes_phenotypes as param
import pandas as pd
from copy import deepcopy

#######################################################################################################################################
print('load data')
#######################################################################################################################################
genemax = int(sys.argv[1])
g9min = int(sys.argv[2])
g9max = int(sys.argv[3])
resolution = int(sys.argv[4])
threshold = int(sys.argv[5])
c0, c1, c2 = 'grey', 'b', 'r' #'#1b9e77', '#d95f02', '#7570b3' #'r', 'b', 'g'
#######################################################################################################################################
string_for_saving_plot=str(genemax)+'_'+str(g9min)+'_'+str(g9max) +'_'+str(resolution) + '_' + str(threshold)
phenotypes=np.load('./biomorphs_map_pixels/phenotypes'+string_for_saving_plot+'.npy')  
max_index_by_site = {pos: phenotypes.shape[pos] - 1 for pos in range(len(phenotypes.shape))}
#######################################################################################################################################
print('load selected values for p0, p1, p2')
#######################################################################################################################################
p0, p1, p2, startgene_ind = param.p0, param.p1, param.p2, param.startgene_ind
assert p0 == phenotypes[startgene_ind]
startgene_str = "".join([str(i) for i in startgene_ind])
#######################################################################################################################################
print('save cpq')
#######################################################################################################################################
NC_filename = './biomorphs_map_pixels/neutral_componentbiomorphs_genotypes_g'+startgene_str+'.npy'
V = get_NC_genotypes(startgene_ind, phenotypes, max_index_by_site)
c_pq, rho = NC_rho_c_pq_biomorphs(phenotypes, startgene_ind)
np.save(NC_filename, V)
np.save('./biomorphs_map_pixels/neutral_componentbiomorphs_cpq_g'+startgene_str+'.npy', c_pq)
np.save('./biomorphs_map_pixels/neutral_componentbiomorphs_rho_g'+startgene_str+'.npy', np.array([rho,]))
print( 'Size of NC of p0: '+str(len(V)))
print( 'bias', c_pq[p1]/c_pq[p2], c_pq[p1], c_pq[p2])
#######################################################################################################################################
print('draw p0, p1, p2')
#######################################################################################################################################
for i, ph in enumerate([p0,p1,p2]):
   f, ax=plt.subplots()
   plot_phenotype(ph, ax, phenotypes, g9min, color=[c0, c1, c2][i], linewidth=7)
   f.savefig('./plots_arrival_frequent/'+['p0', 'p1', 'p2'][i]+'_'+str(ph)+'_phenotypes_'+string_for_saving_plot+'.eps', bbox_inches='tight')
   plt.close()
   del f, ax

#######################################################################################################################################
print('construct graph')
#######################################################################################################################################
G = nx.Graph()
for i, gene in enumerate(V):
   neighbours=neighbours_g_indices_nonperiodic_boundary(gene, max_index_by_site)
   gene_str = "".join([str(i) for i in gene])
   if gene_str not in G.nodes():
      G.add_node(gene_str, pheno='p0')
   for neighbourgene in neighbours:
      ph = phenotypes[tuple(neighbourgene)]
      if ph in [p1, p2, p0]:
         neighbourgene_str = "".join([str(i) for i in neighbourgene])
         if neighbourgene_str not in G.nodes():
            G.add_node(neighbourgene_str, pheno={p0: 'p0', p1: 'p1', p2: 'p2'}[ph])
         G.add_edge(gene_str, neighbourgene_str, weight={p0: 15, p1: 1, p2: 1}[ph])

#######################################################################################################################################
print('plot graph')
#######################################################################################################################################
pos=nx.fruchterman_reingold_layout(G, iterations=500)
nodelist = [deepcopy(n) for n, data in G.nodes(data=True)]
nodecolor_list = [{'p0':c0, 'p1': c1, 'p2': c2}[data['pheno']] for n, data in G.nodes(data=True)]
nodesize_list = [{'p0':4, 'p1': 15, 'p2': 15}[data['pheno']] for n, data in G.nodes(data=True)]
nodes = nx.draw_networkx_nodes(G,pos, nodelist=nodelist, node_color=nodecolor_list, node_size=nodesize_list, linewidths=0) #p0
nx.draw_networkx_edges(G,pos,width=0.11,alpha=0.5, edge_color='k')
plt.tight_layout()
plt.axis('off')
plt.savefig('./plots_arrival_frequent/network_g'+"".join([str(i) for i in startgene_ind])+'_p0_'+str(p0)+'_p1_'+str(p1)+'_p2_'+str(p2)+'.png', bbox_inches='tight', transparent=True, dpi=1000)
#######################################################################################################################################
print('stats on graph')
#######################################################################################################################################
node_vs_pheno=nx.get_node_attributes(G, 'pheno')
connectionsto1=len([(e0, e1) for (e0, e1) in G.edges() if (node_vs_pheno[e0]=='p0' and node_vs_pheno[e1]=='p1') or (node_vs_pheno[e1]=='p0' and node_vs_pheno[e0]=='p1')])
connectionsto2=len([(e0, e1) for (e0, e1) in G.edges() if (node_vs_pheno[e0]=='p0' and node_vs_pheno[e1]=='p2') or (node_vs_pheno[e1]=='p0' and node_vs_pheno[e0]=='p2')])
print( 'bias based on connections: ', float(connectionsto1)/connectionsto2   )
p0_genos_connected_to_p1=len(set([e0 for (e0, e1) in G.edges() if node_vs_pheno[e0]=='p0' and node_vs_pheno[e1]=='p1']+[e1 for (e0, e1) in G.edges() if node_vs_pheno[e0]=='p1' and node_vs_pheno[e1]=='p0']))
p0_genos_connected_to_p2=len(set([e0 for (e0, e1) in G.edges() if node_vs_pheno[e0]=='p0' and node_vs_pheno[e1]=='p2']+[e1 for (e0, e1) in G.edges() if node_vs_pheno[e0]=='p2' and node_vs_pheno[e1]=='p0']))
p0_genos_connected_to_p1p2=len(set([e0 for e0 in G.nodes() if node_vs_pheno[e0]=='p0' and 'p1' in [node_vs_pheno[e1] for e1 in G.neighbors(e0)] and 'p2' in [node_vs_pheno[e1] for e1 in G.neighbors(e0)]]))
print( 'bias based on number of p0 genotypes from which p1/p2 are accessible: ', float(p0_genos_connected_to_p1)/p0_genos_connected_to_p2     )      
print('ratio: connections/portal genotypes for p1:', connectionsto1/p0_genos_connected_to_p1)
print('ratio: connections/portal genotypes for p2:', connectionsto2/p0_genos_connected_to_p2)
print('number of portal genotypes to p1, p2 and both:', p0_genos_connected_to_p1, p0_genos_connected_to_p2, p0_genos_connected_to_p1p2)

##draw transitions from p0 to p2
from functions_for_GP_analysis.biomorph_functions import draw_biomorph_in_subplot
from functions_for_GP_analysis.GPproperties import index_to_gene
from functions_for_GP_analysis.genotype_to_raster import get_biomorphs_phenotype
p0p2_transitions = [(e0, e1) for (e0, e1) in G.edges() if node_vs_pheno[e0]=='p0' and node_vs_pheno[e1]=='p2']
print('examples- changes from p0 to p2', p0p2_transitions)
for i, (g0, g2) in enumerate(p0p2_transitions):
   f, ax=plt.subplots(ncols=4, figsize=(15, 3.5))
   draw_biomorph_in_subplot(index_to_gene(tuple([int(g) for g in g0]), genemax, g9min), ax[0], color='k', linewidth=3)
   draw_biomorph_in_subplot(index_to_gene(tuple([int(g) for g in g2]), genemax, g9min), ax[1], color='k', linewidth=3)
   for j, g in enumerate([g0, g2]):
       array_ph, array_coarsegrained = get_biomorphs_phenotype(index_to_gene(tuple([int(g) for g in g]), genemax, g9min), resolution = 30, test=True)
       im = ax[2 + j].imshow(np.concatenate((array_coarsegrained[:, ::-1], array_coarsegrained), axis=1), origin='lower', cmap='Greys')
       ax[2 + j].set_xlim(0, 30)
       ax[2 + j].set_ylim(0, 30)
       ax[2 + j].axis('equal')
       ax[2 + j].axis('off')
   f.savefig('./plots_arrival_frequent/p0p2_transition_phenotypes_'+string_for_saving_plot+'_' +str(i) +'.eps', bbox_inches='tight')
   plt.close()
   del f, ax

##draw p2 in developmental stages
for i, (g0, g2) in enumerate(p0p2_transitions):
   g2_gene = index_to_gene(tuple([int(g) for g in g2]), genemax, g9min)
   f, ax=plt.subplots(ncols=g2_gene[-1], figsize=(g2_gene[-1] * 3.5, 3.5))
   for j in range(1, g2_gene[-1] + 1):
      draw_biomorph_in_subplot(tuple([g for i, g in enumerate(g2_gene) if i < 8] + [j,]), ax[j-1], color='k', linewidth=3)
   f.savefig('./plots_arrival_frequent/p2_stages_phenotypes_'+string_for_saving_plot+'_' +str(i) +'.eps', bbox_inches='tight')
   plt.close()
   del f, ax 

##robustness
df = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'.csv')
rho  = df_data_to_dict('rho', df)
print('robustness of p1, p2:', rho[p1], rho[p2])

#######################################################################################################################################
print('save initial state of each repetition')
#######################################################################################################################################
from functions_for_GP_analysis.fixation import initialise_population_on_NC
from functools import partial
from multiprocessing import Pool
#############
N, mu, factor_initialisation, no_runs = param.N, param.mu, param.factor_initialisation, param.no_runs
string_pop_param = 'N'+str(int(N))+'_mu'+str(int(10**5 * mu)) + '_' + '_'.join([str(p) for p in [factor_initialisation, p0]])
if not isfile('./data/biomorphs_initial_population'+string_pop_param + string_for_saving_plot+ '_'+str(no_runs -1)+ '.npy'):
   index_and_startgeno_list = [(runindex, tuple(V[index][:])) for runindex, index in enumerate(np.random.choice(len(V), size=no_runs, replace=True))]
   initialisation_function = partial(initialise_population_on_NC, N=N, mu=mu, p0=p0, factor_initialisation=factor_initialisation, phenotypes=phenotypes)
   with Pool(10) as pool:
      result_list = pool.map(initialisation_function, index_and_startgeno_list)
   for index, P_g_current in enumerate(result_list):
      np.save('./data/biomorphs_initial_population'+string_pop_param + string_for_saving_plot+ '_'+str(index)+ '.npy', deepcopy(P_g_current))


