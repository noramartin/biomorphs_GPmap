#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
from matplotlib.gridspec import GridSpec
from functions_for_GP_analysis.biomorph_functions import draw_biomorph_in_subplot
from functions_for_GP_analysis.GPproperties import get_startgeno_indices, neighbours_g_indices_nonperiodic_boundary, gene_to_index, index_to_gene
from functions_for_GP_analysis.neutral_component import find_NCs_specific_pheno_with_genotype_set_info
import pandas as pd
from os.path import isfile
from functools import partial
from multiprocessing import Pool
import definition_special_genotypes_phenotypes as param
from functions_for_GP_analysis.genotype_to_raster import get_biomorphs_phenotype
from copy import deepcopy


def save_NC_versusNCneighbours(NCarray, GPmapdef):
   filename_NC_map_network = './biomorphs_map_pixels/neutral_component_network'+GPmapdef+'.csv'
   print( filename_NC_map_network)
   if isfile(filename_NC_map_network):
      df_NCnetwork = pd.read_csv(filename_NC_map_network)
   else:   
      print( 'mutational neighbourhood of each NC')
      L = NCarray.ndim
      max_int_for_pos = {pos: NCarray.shape[pos]-1 for pos in range(L)}
      NC_vs_neighbours = {NC: set([]) for NC in np.arange(1, np.max(NCarray) + 1)} #counts the number of times each relevant pair of phenoypes appears in the sample
      for genotype, NC in np.ndenumerate(NCarray):
         neighbours = neighbours_g_indices_nonperiodic_boundary(tuple(genotype), max_int_for_pos)
         for neighbourgeno in neighbours:
            NC_vs_neighbours[NC].add(NCarray[tuple(neighbourgeno)])
      for NC, neighbours_list in NC_vs_neighbours.items():
         NC_vs_neighbours[NC] = '_'.join([str(n) for n in neighbours_list])
      df_NCnetwork = pd.DataFrame.from_dict({'NC': [NC for NC in NC_vs_neighbours.keys()], 
                                            'neighbour NC list': [NC_vs_neighbours[NC][:] for NC in NC_vs_neighbours.keys()]})
      df_NCnetwork.to_csv(filename_NC_map_network)
   return df_NCnetwork

def find_all_phenotype_neighbours(startpheno, phenotypes):
   genes_same_string=np.argwhere(phenotypes==startpheno)
   set_pheno_neighbours=set([])
   for gene in genes_same_string:
      set_pheno_neighbours.update([phenotypes[tuple(g)] for g in neighbours_g_indices_nonperiodic_boundary(gene, np.array(phenotypes.shape)-1)])
   return list(set_pheno_neighbours)

def Dijkstra_shortest_path(startNC, endNC, df_NCnetwork):
   unfinished_nodes = list(range(np.min(df_NCnetwork['NC'].tolist()), np.max(df_NCnetwork['NC'].tolist())+1))
   max_dist = np.max(df_NCnetwork['NC'].tolist())+5
   distances = {n: max_dist for n in unfinished_nodes}
   distances[startNC] = 0
   previous_nodes = {}
   NC_vs_neighbours = {row['NC']: [int(n) for n in row['neighbour NC list'].split('_')] for rowindex, row in df_NCnetwork.iterrows()}
   del df_NCnetwork
   while len(unfinished_nodes)>0:
      next_node = min(unfinished_nodes, key=distances.get)
      if next_node == endNC or distances[next_node] == distances[endNC]:
        break                  
      unfinished_nodes.remove(next_node)  
      neighbours_of_current_node = NC_vs_neighbours[next_node] 
      for neighbor in neighbours_of_current_node:
         new_dist = distances[next_node]+1  
         if new_dist < distances[neighbor]:
            distances[neighbor] = new_dist
            previous_nodes[neighbor] = next_node
      del neighbours_of_current_node, neighbor, new_dist, next_node
   #traceback to get shortest path
   shortest_path_list = [endNC]
   current_node = endNC
   while current_node != startNC:
      current_node = previous_nodes[current_node]
      shortest_path_list.append(current_node)
   return shortest_path_list[::-1]

############################################################################################################
## identify neutral components
############################################################################################################

def find_all_NCs_parallel(GPmap, max_index_by_site, phenotype_list_forNC, GPmapdef=None):
   #find individual neutral components for all phenotypes in GP map;
   #saves/retrieves information if GPmapdef given and files present; otherwise calculation from scratch
   #code adapted from stability project
   filename_NC_map = './biomorphs_map_pixels/neutral_components'+GPmapdef+'.npy'
   print( filename_NC_map)
   if isfile(filename_NC_map):
      seq_vs_NCindex_array = np.load(filename_NC_map)
   else:
      print( 'NC file not found')
      seq_vs_NCindex_array = np.zeros_like(GPmap, dtype='uint32')
      NCindex_global_count = 1 #such that 0 remains for unfinished part
      while len(phenotype_list_forNC):
         print( 'number of phenotypes for NC calculation', len(phenotype_list_forNC))
         phenotypes_this_loop, pheno_is_in_loop, total_no_genotypes = [], np.zeros(np.max(GPmap) + 1, dtype='int8'), 0
         for ph in phenotype_list_forNC:
            if total_no_genotypes + ph_vs_N[ph]  < 5 * 10**5 or len(phenotypes_this_loop) < 1: # this cut-off is responsible for the trade-off between speed and memory
               phenotypes_this_loop.append(ph)
               pheno_is_in_loop[ph] = 1
               total_no_genotypes += ph_vs_N[ph] 
            if total_no_genotypes >= 5 * 10**5:
               break
         if len(phenotypes_this_loop) == 1: # method for large NCs
            ph = phenotypes_this_loop[0]
            for genotype_indices, p in np.ndenumerate(GPmap):
               if p == ph and seq_vs_NCindex_array[genotype_indices] < 0.5:
                  seq_vs_NCindex_array = neutral_component_largepheno(genotype_indices, GPmap, seq_vs_NCindex_array, NCindex_global_count, max_index_by_site)
                  NCindex_global_count += 1
            phenotype_list_forNC.remove(ph)
         else:
            print( 'find initial genotypes for the following number of phenotypes', len(phenotypes_this_loop))
            ph_vs_geno_list = {ph: [] for ph in phenotypes_this_loop}
            for genotype_indices, ph in np.ndenumerate(GPmap):
               if pheno_is_in_loop[ph] > 0.5:
                  ph_vs_geno_list[ph].append(tuple(deepcopy(genotype_indices)))
            print( 'start NC search')
            find_NC_correct_const_arguments = partial(find_NCs_specific_pheno_with_genotype_set_info, phenotype_vs_genotypelist = ph_vs_geno_list, GPmap=GPmap, max_index_by_site=max_index_by_site)
            pool = Pool()
            NC_per_pheno_list = pool.map(find_NC_correct_const_arguments, phenotypes_this_loop) 
            pool.close()
            pool.join()
            for phenoindex, pheno in enumerate(phenotypes_this_loop):
               map_to_global_NCindex = {NCindex: NCindex_global_count + NCindex for NCindex in set([v for v in NC_per_pheno_list[phenoindex].values()])}
               NCindex_global_count = max(map_to_global_NCindex.values()) + 1
               for seq, NCindex in NC_per_pheno_list[phenoindex].items():
                  assert map_to_global_NCindex[NCindex] <  4294967295
                  seq_vs_NCindex_array[seq] = map_to_global_NCindex[NCindex]
            del NC_per_pheno_list, map_to_global_NCindex
            for ph in phenotypes_this_loop:
               phenotype_list_forNC.remove(ph)
            del phenotypes_this_loop, pheno_is_in_loop, ph_vs_geno_list
      np.save(filename_NC_map, seq_vs_NCindex_array)
   return seq_vs_NCindex_array



def neutral_component_largepheno(g0, phenotypes, NCarray, NCindex_for_array, max_index_by_site):
   #using the algorithm from Gruener et al. (1996)
   U=np.zeros_like(phenotypes, dtype='int8') #zero means not in set - initialise U as both array and list for faster lookup
   U_list=[]
   pheno=int(phenotypes[tuple(g0)])
   U[tuple(g0)]=1
   U_list.append(tuple(g0))
   while len(U_list)>0: #while there are elements in the unvisited list
      g1 = deepcopy(U_list[0] )
      neighbours = neighbours_g_indices_nonperiodic_boundary(g1, max_index_by_site)
      assert len(neighbours) == 2 * len(g1) - g1.count(0) - list(max_index_by_site - g1).count(0)
      for g2 in neighbours:
         ph2 = int(phenotypes[tuple(g2)])
         if ph2==pheno and U[tuple(g2)]==0 and abs(NCarray[tuple(g2)] - NCindex_for_array) > 0.5:
            U[tuple(g2)]=1
            U_list.append(tuple(g2))
      U[tuple(g1)]=0
      NCarray[tuple(g1)]=NCindex_for_array #visited list
      U_list.remove(tuple(g1))
   return NCarray


def get_all_genes_given_pheno(phenotypes, ph, ph_is_rare=True):
   if ph_is_rare:
      return [tuple(deepcopy(g)) for g in np.argwhere(phenotypes==ph)]
   else:
      return [tuple(deepcopy(g)) for g, pheno in np.ndenumerate(phenotypes) if ph == pheno]
      

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################


genemax = int(sys.argv[1])
g9min = int(sys.argv[2])
g9max = int(sys.argv[3])
resolution = int(sys.argv[4])
threshold = int(sys.argv[5])
string_for_saving_plot=str(genemax)+'_'+str(g9min)+'_'+str(g9max) +'_'+str(resolution) + '_'+str(threshold)
phenotypes=np.load('./biomorphs_map_pixels/phenotypes'+string_for_saving_plot+'.npy')  
################################################################################################################################################################################################################################
initial_genotype_not_in_indices = param.initial_genotype_not_in_indices
final_genotype_not_in_indices = param.final_genotype_not_in_indices
####################################################################
start_genotype_indices = gene_to_index(initial_genotype_not_in_indices, genemax, g9min)
final_genotype_indices = gene_to_index(final_genotype_not_in_indices, genemax, g9min)
start_phenotype = phenotypes[tuple(start_genotype_indices)]
final_phenotype = phenotypes[tuple(final_genotype_indices)]
print('start and end', start_phenotype, final_phenotype)

################################################################################################################################################################################################################################
### load neutral set size
################################################################################################################################################################################################################################
df_GPmap_properties = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'.csv')
ph_vs_N = {ph: N for ph, N in zip(df_GPmap_properties['phenotype int'].tolist(), df_GPmap_properties['neutral set size'].tolist())}
del df_GPmap_properties
################################################################################################################################################################################################################################
### compute/load NCs
################################################################################################################################################################################################################################
phenotype_list_forNC = [ph for ph, f in ph_vs_N.items()] # if  9.5 < f < 10**4]
neutral_component_array = find_all_NCs_parallel(phenotypes,  max_index_by_site = np.array(phenotypes.shape) - 1, phenotype_list_forNC = phenotype_list_forNC, GPmapdef=string_for_saving_plot)
################################################################################################################################################################################################################################
shortest_path_filename = './biomorphs_map_pixels/shortest_phenotype_path ' +string_for_saving_plot+'.csv'

if not isfile(shortest_path_filename):
   ################################################################################################################################################################################################################################
   ### compute/load NCs
   ################################################################################################################################################################################################################################
   df_NCnetwork = save_NC_versusNCneighbours(neutral_component_array, GPmapdef=string_for_saving_plot)
   ################################################################################################################################################################################################################################
   ### find NCindices of start and end phenotype
   ################################################################################################################################################################################################################################
   final_NCindex_list = list(set([neutral_component_array[tuple(g)] for g in get_all_genes_given_pheno(phenotypes, final_phenotype, ph_is_rare=True)]))
   start_NCindex_list = list(set([neutral_component_array[tuple(g)] for g in get_all_genes_given_pheno(phenotypes, start_phenotype, ph_is_rare=False)]))
   print( 'startNCs', start_NCindex_list)
   print( 'endNCs', final_NCindex_list)
   ################################################################################################################################################################################################################################
   ### find shortest paths in NC space
   ################################################################################################################################################################################################################################
   shortest_path_list = []
   for startNCindex in start_NCindex_list:
       for endNCindex in final_NCindex_list:
          nodes_along_one_shortest_path = Dijkstra_shortest_path(endNCindex, startNCindex, df_NCnetwork)[::-1]
          shortest_path_list.append('_'.join([str(n) for n in nodes_along_one_shortest_path]))
          ### test that shortest path is valid
          for i in range(len(nodes_along_one_shortest_path)-1):
            assert nodes_along_one_shortest_path[i] in find_all_phenotype_neighbours(nodes_along_one_shortest_path[i+1], neutral_component_array) 
            assert  nodes_along_one_shortest_path[i+1] in find_all_phenotype_neighbours(nodes_along_one_shortest_path[i], neutral_component_array)

   df_shortest_path = pd.DataFrame.from_dict({'NCs along shortest path':  shortest_path_list, 'NC combination': np.arange(len(shortest_path_list))})
   df_shortest_path.to_csv(shortest_path_filename)

################################################################################################################################################################################################################################
### prepae for drawing
################################################################################################################################################################################################################################
df_shortest_path = pd.read_csv(shortest_path_filename)
nodes_along_one_shortest_path = min(deepcopy([[int(NCindex) for NCindex in path.split('_')] for path in df_shortest_path['NCs along shortest path'].tolist()]), key=len)
startNC = nodes_along_one_shortest_path[0]
#df_NCnetwork = save_NC_versusNCneighbours(neutral_component_array, GPmapdef=string_for_saving_plot)
#print('evolvability of NC', startNC, [len([phenotypes[get_startgeno_indices(neutral_component_array, int(n))] for n in row['neighbour NC list'].split('_')]) for rowindex, row in df_NCnetwork.iterrows() if row['NC'] == startNC][0])
#del df_NCnetwork
################################################################################################################################################################################################################################
### run tests on NCs in general
################################################################################################################################################################################################################################
for NCindex in nodes_along_one_shortest_path:
   #test NCs in general
   genotypes_network = [tuple(g) for g in get_all_genes_given_pheno(neutral_component_array, NCindex, ph_is_rare=False)]
   if len(genotypes_network) > 1:
      g1index, g2index = np.random.choice(len(genotypes_network), 2, replace=False)
      g1, g2 =  index_to_gene(tuple(deepcopy(genotypes_network[g1index])), genemax, g9min), index_to_gene(tuple(deepcopy(genotypes_network[g2index])), genemax, g9min)
      assert get_biomorphs_phenotype(g1,resolution=resolution, threshold=threshold) == get_biomorphs_phenotype(g2,resolution=resolution, threshold=threshold)  
#print('phenotypes on path', [phenotypes[get_startgeno_indices(neutral_component_array, NCindex)] for NCindex in nodes_along_one_shortest_path])
################################################################################################################################################################################################################################
### draw shortest path from dot to beetle
################################################################################################################################################################################################################################
no_plots=len(nodes_along_one_shortest_path)
ncols = 3
nrows = int(np.ceil(no_plots/float(ncols)))
fig = plt.figure(figsize=(ncols*3, nrows))
gs = GridSpec(ncols=2*ncols + 1, nrows=nrows, width_ratios=[1] * ncols + [0.5] + [1] * ncols)
for index, NCindex in enumerate(nodes_along_one_shortest_path):
   ax_plot = fig.add_subplot(gs[index//ncols, index%ncols])
   gene = get_startgeno_indices(neutral_component_array, NCindex)
   if index == 0:
      ax_plot.scatter(0, 0, c='k', s=5)
      ax_plot.set_aspect('equal', 'box')
      ax_plot.axis('off')
   else:
      draw_biomorph_in_subplot(index_to_gene(gene, genemax, g9min), ax_plot, color='k', linewidth=1)
   if index == 0:
      assert phenotypes[gene] == phenotypes[start_genotype_indices]
   elif index == len(nodes_along_one_shortest_path) - 1:
      assert phenotypes[gene] == phenotypes[final_genotype_indices]
axbig = fig.add_subplot(gs[:, 1 + ncols:])
line_of_genotypes_indices = [get_startgeno_indices(neutral_component_array, NCindex) for NCindex in nodes_along_one_shortest_path]
axbig.plot(np.arange(len(line_of_genotypes_indices)), [ph_vs_N[phenotypes[tuple(gene_index[:])]] for gene_index in line_of_genotypes_indices], c='grey', lw=1.5)
axbig.set_yscale('log')
axbig.set_ylabel('neutral set size')
axbig.set_xlabel("number of phenotypic changes since start")
xmin, xmax = axbig.get_xlim()
axbig.set_xticks(list(np.arange(0, xmax)))
axbig.set_xticklabels([str(int(i)) for i in list(np.arange(0, xmax))])
fig.tight_layout()
fig.savefig('./plots_beetle/Dec18_path_to_Dawkins_beetle_'+string_for_saving_plot+'_phenotypespace.eps')
plt.close()
del fig, gs



