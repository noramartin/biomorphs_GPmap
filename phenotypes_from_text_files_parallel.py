#!/usr/bin/env python3
"""
Calculate the phenotypes and save them in an 8-dim array
-position in array given by [g1+genemax,g2+genemax,g3+genemax,g4+genemax,g5+genemax,g6+genemax,g7+genemax,g8+genemax]
-phenotypes represented by numbers
"""


import numpy as np
import sys
import pandas as pd
from functions_for_GP_analysis.GPproperties import gene_to_index, mirror_x
from itertools import product
from os.path import isfile
from copy import deepcopy

def compress_binary_string(binstr):
   """returns tuple which describes concatenation of binary strings binstr1+binstr2: (number of preceding zeroes, corresponding decimal number)"""
   number_decimal = int(binstr,2) #convert binary string into decimal string
   number_zeros_beginning = 0
   for i in binstr:
      if int(i) == 0:
         number_zeros_beginning += 1
      else:
         break
   return (int(number_zeros_beginning), int(number_decimal))

assert compress_binary_string('00010') == (3, 2)
assert compress_binary_string('00011') == (3, 3)
assert compress_binary_string('0011') == (2, 3)
assert compress_binary_string('001010') == (2, 10)
###########################################################################################################################################
genemax=int(sys.argv[1])
g9min=int(sys.argv[2])
g9max=int(sys.argv[3])
resolution=int(sys.argv[4])
threshold=int(sys.argv[5])


dim_array = 2*genemax+1
phenotypes = np.empty([dim_array,dim_array,dim_array,dim_array,dim_array,dim_array,dim_array,dim_array, g9max-g9min+1], dtype='uint32')
phenotypes_number_dict = {}
number_pheno=0
progress_count = 0
vectorgene_values = [g for g in range(-genemax,genemax+1)]
filename = './biomorphs_map_pixels/phenotypes'+str(genemax)+'_'+str(g9min)+'_'+str(g9max)+'_'+str(resolution)+'_'+str(threshold)+'.npy'
if not isfile(filename):
   for g9 in range(g9min,g9max+1):
      for g1 in range(0, genemax + 1):
         #need to read in .txt file
         branches_file = open('./biomorphs_map_pixels/binary_string_phenotypes_genemax'+str(genemax)+'_g9_'+str(g9)+'_g1_'+str(g1) +'_'+str(resolution) + '_'+str(threshold) +'.txt')
         branches_lines = branches_file.read()
         branches_data = branches_lines.split()
         assert len(branches_data) == int(dim_array**7)
         print('order of recursion: ', g9, 'g1', g1)
         #quantities needed for the loop
         list_all_genotypes = [tuple([g1,] + [g for g in vector_genes] + [g9,]) for vector_genes in product(vectorgene_values, repeat=7)]
         for list_index, gene_new in enumerate(list_all_genotypes):
            phenotype_string = deepcopy(branches_data[list_index])         
            phenotype_tuple = compress_binary_string(phenotype_string)
            #make a dictionary mapping phenotypes to a number (their rank)
            try:
               ph_integer = phenotypes_number_dict[phenotype_tuple]
            except KeyError:
               phenotypes_number_dict[phenotype_tuple] = int(number_pheno)
               number_pheno += 1
               ph_integer = phenotypes_number_dict[phenotype_tuple]
            phenotypes[gene_to_index(gene_new, genemax, g9min)] = ph_integer #enter the phenotype into the array
            phenotypes[gene_to_index(mirror_x(gene_new), genemax, g9min)] = ph_integer #enter the phenotype into the array
            progress_count += 1
            assert number_pheno < 4294967295 #largest integer allowed in 'uint32' array
            if progress_count%500000==0:
               print( str(progress_count)+'th genotype evaluated')
         del branches_file, branches_data, branches_lines
                            


   np.save(filename, phenotypes)
   np.save('./biomorphs_map_pixels/phenotypes_number_dict'+str(genemax)+'_'+str(g9min)+'_'+str(g9max)+'_'+str(resolution)+'_'+str(threshold)+'.npy', phenotypes_number_dict)

