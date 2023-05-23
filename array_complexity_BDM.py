#!/usr/bin/env python3
import numpy as np
import sys
from functions_for_GP_analysis.GPproperties import tuple_to_binary_string, expand_array
import pandas as pd
from os.path import isfile
from os import cpu_count
from multiprocessing import Pool
from copy import deepcopy
from functools import partial
from pybdm import BDM, PartitionCorrelated


def BDM_complexity_from_tuple_notation(tuple_ph, resolution):
   return bdm.bdm(np.array(expand_array(tuple_to_binary_string(tuple_ph, resolution)), dtype='int'))

  

genemax = int(sys.argv[1])
g9min = int(sys.argv[2])
g9max = int(sys.argv[3])
resolution = int(sys.argv[4])
threshold = int(sys.argv[5])

bdm = BDM(ndim=2, partition=PartitionCorrelated)


string_for_saving_plot=str(genemax)+'_'+str(g9min)+'_'+str(g9max) +'_'+str(resolution)+'_'+str(threshold)
tuple_vs_int_list = np.load('./biomorphs_map_pixels/phenotypes_number_dict'+string_for_saving_plot+'.npy', allow_pickle=True).item()
ph_list_tuples, ph_list_ints = zip(*[(deepcopy(tuple_ph), int_ph) for tuple_ph, int_ph in tuple_vs_int_list.items()])

####################################################################################
print('check format of strings')
####################################################################################
for tuple_ph in ph_list_tuples:
  assert len(tuple_to_binary_string(tuple_ph, resolution)) == resolution**2//2 
####################################################################################
print('get complexity and save')
####################################################################################
filename_complexitydata = './biomorphs_map_pixels/block_decomp_arraycomplexitydata'+string_for_saving_plot+'_correlated.csv'
parallel_function = partial(BDM_complexity_from_tuple_notation, resolution=resolution)
if not isfile(filename_complexitydata):
   with Pool(cpu_count()) as p:
       array_complexity_list_pool_output = p.map(parallel_function, deepcopy(ph_list_tuples), chunksize=10**3)
   del ph_list_tuples
   array_complexity_list = np.array([c for c in array_complexity_list_pool_output])
   df = pd.DataFrame.from_dict({'phenotype integer': ph_list_ints, 'array complexity': array_complexity_list})
   df.to_csv(filename_complexitydata)

