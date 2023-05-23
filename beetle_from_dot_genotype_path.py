import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
from functions_for_GP_analysis.biomorph_functions import draw_biomorph_in_subplot
from functions_for_GP_analysis.GPproperties import gene_to_index, index_to_gene
import definition_special_genotypes_phenotypes as param


def find_path(start_genotype_indices, final_genotype_indices, phenotypes):
   path_of_genotypes = [deepcopy(list(start_genotype_indices))]
   current_genotype = deepcopy(list(start_genotype_indices))
   genotypes_final_phenotype = [tuple(g) for g in np.argwhere(phenotypes == phenotypes[tuple(final_genotype_indices)])]
   genotypes_final_phenotype_vs_dist_initial = {tuple(deepcopy(g)): sum([abs(i - j) for (i, j) in zip(g, current_genotype)]) for g in genotypes_final_phenotype}
   final_genotype_indices = tuple(min(genotypes_final_phenotype, key = genotypes_final_phenotype_vs_dist_initial.get))
   while phenotypes[tuple(current_genotype)] != phenotypes[tuple(final_genotype_indices)]:
      different_indices_that_change_phenotype = [i for i in range(9) if current_genotype[i]!=final_genotype_indices[i]]
      index_to_mutate = np.random.choice(different_indices_that_change_phenotype)
      current_genotype[index_to_mutate] += np.sign(final_genotype_indices[index_to_mutate]-current_genotype[index_to_mutate])
      path_of_genotypes.append(deepcopy(current_genotype))
   return path_of_genotypes


np.random.seed(0)
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
genemax = int(sys.argv[1])
g9min = int(sys.argv[2])
g9max = int(sys.argv[3])
resolution = int(sys.argv[4])
threshold = int(sys.argv[5])

total_number_runs=50
string_for_saving_plot=str(genemax)+'_'+str(g9min)+'_'+str(g9max) +'_'+str(resolution) + '_'+str(threshold)
phenotypes=np.load('./biomorphs_map_pixels/phenotypes'+string_for_saving_plot+'.npy')  
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################################################################################################################################
initial_genotype_not_in_indices = param.initial_genotype_not_in_indices
final_genotype_not_in_indices = param.final_genotype_not_in_indices
####################################################################
start_genotype_indices = gene_to_index(initial_genotype_not_in_indices, genemax, g9min)
final_genotype_indices = gene_to_index(final_genotype_not_in_indices, genemax, g9min)
start_phenotype = phenotypes[tuple(start_genotype_indices)]
final_phenotype = phenotypes[tuple(final_genotype_indices)]

################################################################################################################################################################################################################################
### get data
################################################################################################################################################################################################################################
no_run_vs_glist = {}
for no_run in range(total_number_runs):
   glist = find_path(start_genotype_indices, final_genotype_indices, phenotypes)
   no_run_vs_glist[no_run] = deepcopy(glist)

################################################################################################################################################################################################################################
### without neutral set sizes
################################################################################################################################################################################################################################
for no_run in range(total_number_runs):

   line_of_genotypes_indices = deepcopy(no_run_vs_glist[no_run])
   print(no_run, len(line_of_genotypes_indices))
   #####plot
   startindex = min([i for i, g in enumerate(line_of_genotypes_indices) if phenotypes[tuple(deepcopy(g))] != phenotypes[tuple(line_of_genotypes_indices[0])]]) - 1
   line_of_genotypes_indices = line_of_genotypes_indices[startindex:]
   no_genos=len(line_of_genotypes_indices)
   no_plots = no_genos #len([phenotypes[tuple(deepcopy(g))] for i, g in enumerate(line_of_genotypes_indices) if (i == 0 or phenotypes[tuple(deepcopy(g))] != phenotypes[tuple(line_of_genotypes_indices[i - 1])])])
   fig, ax = plt.subplots(ncols=no_plots, figsize=(no_plots*1.75, 1))
   previous_phenotype = -1
   index = 0
   for gene_indices in line_of_genotypes_indices:
      if True: #phenotypes[tuple(gene_indices)] != previous_phenotype:
         gene = index_to_gene(gene_indices, genemax, g9min)
         if index == 0:
            ax[index].scatter(0, 0, c='k', s=10)
            ax[index].set_aspect('equal', 'box')
            ax[index].axis('off')
         else:
            draw_biomorph_in_subplot(gene, ax[index], color='k', linewidth=1)
         index += 1
         previous_phenotype = phenotypes[tuple(gene_indices)]
   fig.savefig('./plots_beetle/Dec18_path_to_Dawkins_beetle_'+string_for_saving_plot+'_ver'+str(no_run)+'_genotypespace_biomorphs_only.eps')
   plt.close()
   del fig, ax
 





