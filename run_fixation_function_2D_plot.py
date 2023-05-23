#!/usr/bin/env python
import numpy as np
import sys
from os.path import isfile
from functions_for_GP_analysis.fixation import *
import definition_special_genotypes_phenotypes as param


#s1_list=[0.001*4**i for i in range(5)]
#s1s2_list=[(s1, s2) for s1 in s1_list for s2 in s1_list]
s1_list = [s1 for s1 in param.s1_list]
s1s2_list=[(s1, s2) for s1 in s1_list for s2 in s1_list if s2 > s1]
#######################################################################################################################################
##input: population properties
#######################################################################################################################################
N = param.N
mu = param.mu
Tmax = 10**8 #no of generations after which classed as failed
no_runs = param.no_runs
factor_initialisation = param.factor_initialisation
s_index = int(sys.argv[7])
ncpus = int(sys.argv[6])
s1, s2 = s1s2_list[s_index]
print( 's1: '+str(s1))
print( 's2: '+str(s2))
#assert s2 > s1
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
string_pop_param = 'N'+str(int(N))+'_logTmax'+str(int(np.log10(Tmax)))+'_mu'+str(int(10**5 * mu))+'s1'+str(int(10**4 * s1))+'s2'+str(int(10**4 * s2))+'runs_'+str(int(no_runs)) + '_' + '_'.join([str(p) for p in [factor_initialisation, p0, p1, p2]])
#######################################################################################################################################
if not isfile('./data/biomorphs_fixation_'+string_pop_param + string_for_saving_plot+'.npy'):
   fixation_result, p1_found, p2_found = fixation_probability(N, mu, Tmax, no_runs, s1, s2, p0, p1, p2, str_startgene, string_for_saving_plot, factor_initialisation=factor_initialisation, ncpus=ncpus)
   #######################################################################################################################################
   print( 'fixations: ')
   print( fixation_result)
   print( 'p1 found: ')
   print( p1_found)
   print( 'p2 found: ')
   print( p2_found)
#######################################################################################################################################
#######################################################################################################################################
from functions_for_GP_analysis.long_run_two_peaked_fitness import run_simulation_until_Tmax_and_save

no_runs, Tmax = 100, 10**6
#s1, s2 = 0.21, 0.41
string_pop_param = 'N'+str(int(N))+'_logTmax'+str(int(np.log10(Tmax)))+'_mu'+str(int(10**5 * mu))+'s1'+str(int(10**4 * s1))+'s2'+str(int(10**4 * s2))+'runs_'+str(int(no_runs)) + '_' + '_'.join([str(p) for p in [factor_initialisation, p0, p1, p2]])
if not isfile('./data/longrun_biomorphs_fixation_'+string_pop_param + string_for_saving_plot+'.txt'):
    run_simulation_until_Tmax_and_save(N, mu, Tmax, no_runs, s1, s2, p0, p1, p2, str_startgene, string_for_saving_plot, factor_initialisation=factor_initialisation, ncpus=ncpus)




