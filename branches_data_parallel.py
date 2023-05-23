#!/usr/bin/env python3
"""
Calculate the phenotypes and save them in an 8-dim array
-position in array given by [g1+genemax,g2+genemax,g3+genemax,g4+genemax,g5+genemax,g6+genemax,g7+genemax,g8+genemax, g9+g9min]
-phenotypes represented by numbers, such  that each integer value up to a fixed maximum corresponds to one phenotype
"""


import numpy as np
from functions_for_GP_analysis.genotype_to_raster import get_biomorphs_phenotype
import sys
from multiprocessing import Pool
from itertools import product
from functools import partial
from os.path import isfile
from os import cpu_count
from copy import deepcopy


print(sys.version)
print('number cpu count', cpu_count())

genemax=int(sys.argv[1])
g9min=int(sys.argv[2])
g9max=int(sys.argv[3])
resolution= int(sys.argv[4])
threshold = int(sys.argv[5])
assert threshold > 1 #1/threshold is the fraction of the pixel edge length that is used as a cut-off when binarising the pixel array


vectorgene_values = [g for g in range(-genemax, genemax+1)]
GPfunction = partial(get_biomorphs_phenotype, resolution=resolution, threshold = threshold)
############################################
for g9 in range(g9min, g9max+1):
   for g1 in range(0, genemax + 1):
      filename = './biomorphs_map_pixels/binary_string_phenotypes_genemax'+str(genemax)+'_g9_'+str(g9)+'_g1_'+str(g1) +'_'+str(resolution) + '_'+str(threshold)+'.txt'
      if not isfile(filename):
         print( 'g9: ', g9, 'g1:', g1)
         list_all_genotypes = [tuple([g1,] + [g for g in vector_genes] + [g9,]) for vector_genes in product(vectorgene_values, repeat=7)]
         #phenotype_list = [GPfunction(g) for g in list_all_genotypes]
         with Pool(cpu_count()) as pool:
            phenotype_list = pool.map(GPfunction, list_all_genotypes)
         for g in list_all_genotypes:
            del g
         del list_all_genotypes
         max_stringlength = max([len(p) for p in phenotype_list])
         assert max_stringlength == resolution**2 //2
         np.savetxt(filename, deepcopy(phenotype_list), fmt='%1.'+str(max_stringlength)+'s', delimiter='\n')
         for p in phenotype_list:
            del p 
         del phenotype_list

