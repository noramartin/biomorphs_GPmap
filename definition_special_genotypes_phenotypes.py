import numpy as np
from functions_for_GP_analysis.GPproperties import random_startgene, index_to_gene

def generate_genotype_for_line(phenotypes, genemax, g9min, g9max):
	ph_line = phenotypes[tuple([0,] * 9)]
	while True:
		g = random_startgene(genemax, g9min, g9max)
		if phenotypes[tuple(g)] == ph_line:
			return index_to_gene(tuple(g[:]), genemax, g9min)

############################################################
## suggestions for special phenotype for direct path
############################################################
# genemax, g9min, g9max, resolution, threshold = 3, 1, 8, 30, 5
# string_for_saving_plot=str(genemax)+'_'+str(g9min)+'_'+str(g9max) +'_'+str(resolution) + '_'+str(threshold)
# phenotypes=np.load('./biomorphs_map_pixels/phenotypes'+string_for_saving_plot+'.npy')

# print('suggestion for initial genotype (not in index notation): ', generate_genotype_for_line(phenotypes, genemax, g9min, g9max))
############################################################
## special phenotype for direct path
############################################################
final_genotype_not_in_indices = (-1, 2, 1, 2, -3, -1, -1, 2, 6)
initial_genotype_not_in_indices = (1, -2, -2, 0, 2, 0, -3, 3, 1) 
############################################################
## special phenotypes for two-peaked landscape
############################################################
p0, p1, p2 = 905, 4179, 92259 
startgene_ind =  (1, 3, 5, 1, 3, 3, 1, 1, 2)
no_runs = 10**3 
N = 500 
mu = 0.0001 
factor_initialisation = 10
s1_list = [0.02] + [s * 0.01 for s in range(5, 50, 5)]