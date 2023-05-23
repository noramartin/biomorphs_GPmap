#!/usr/bin/env python
import numpy as np
from multiprocessing import Pool
from functools import partial
try:
   from neutral_component import get_NC_genotypes
except ModuleNotFoundError:
   from .neutral_component import get_NC_genotypes
import random
import time
from copy import deepcopy


def fixation_probability(N, mu, Tmax, no_runs, s1, s2, p0, p1, p2, str_startgene, string_for_saving_plot, factor_initialisation, ncpus=15):
   global phenotypes
   phenotypes = np.load('./biomorphs_map_pixels/phenotypes'+string_for_saving_plot+'.npy')  
   #input: Gp map and properties
   string_pop_param_for_initial_condition = 'N'+str(int(N))+'_mu'+str(int(10**5 * mu)) + '_' + '_'.join([str(p) for p in [factor_initialisation, p0]])
   filename_for_initial_condition_list = ['./data/biomorphs_initial_population'+string_pop_param_for_initial_condition + string_for_saving_plot+ '_'+str(index)+ '.npy' for index in range(no_runs)]
   #initialise 
   discovery_times_p1_p2 = np.zeros((no_runs, 2), dtype='int32')
   fixation = np.zeros(no_runs, dtype='int16') #fixation results: enter 0 for no fixation, 1 for p1 fixation and 2 for p2
   p1_found = np.zeros(no_runs, dtype='int16')
   p2_found = np.zeros(no_runs, dtype='int16')
   ########evolution on this map  
   fixation_function_with_param = partial(run_simulation_until_fixation, N=N, mu=mu, Tmax=Tmax, s1=s1, s2=s2, p0=p0, p1=p1, p2=p2, factor_initialisation=factor_initialisation) #, phenotypes=phenotypes)
   #print [fixation_function_with_param(a) for a in startgeno_list[:3]]
   pool = Pool(processes=ncpus)
   result_list = pool.map(fixation_function_with_param, filename_for_initial_condition_list)
   pool.close()
   pool.join()
   for run in range(no_runs):
       fixation_thisrun, p1_found_thisrun, p2_found_thisrun, discoverytime_p1, discoverytime_p2 = result_list[run] 
       print('p1_found_thisrun, p2_found_thisrun', p1_found_thisrun, p2_found_thisrun)
       fixation[run] = fixation_thisrun
       p1_found[run] = p1_found_thisrun
       p2_found[run] = p2_found_thisrun
       discovery_times_p1_p2[run,0] = discoverytime_p1
       discovery_times_p1_p2[run,1] = discoverytime_p2
   ### save at the end
   string_pop_param = 'N'+str(int(N))+'_logTmax'+str(int(np.log10(Tmax)))+'_mu'+str(int(10**5 * mu))+'s1'+str(int(10**4 * s1))+'s2'+str(int(10**4 * s2))+'runs_'+str(int(no_runs)) + '_' + '_'.join([str(p) for p in [factor_initialisation, p0, p1, p2]])
   np.save( './data/biomorphs_fixation_'+string_pop_param + string_for_saving_plot+'.npy', fixation)
   np.save( './data/biomorphs_p1found_'+string_pop_param + string_for_saving_plot+'.npy', p1_found)
   np.save( './data/biomorphs_p2found_'+string_pop_param + string_for_saving_plot+'.npy', p2_found)
   np.save( './data/biomorphs_discoverytimes_'+string_pop_param + string_for_saving_plot+'.npy', discovery_times_p1_p2)
   return fixation, p1_found, p2_found
   

def run_simulation_until_fixation(filename_for_initial_condition, N, mu, Tmax, s1, s2, p0, p1, p2, factor_initialisation): #, phenotypes):
   def fitness(phenotype):
      if int(phenotype) == int(p0):
         return 1.0
      elif int(phenotype) == int(p1):
         return 1 + s1
      elif int(phenotype) == int(p2):
         return 1 + s2
      else:
         return 0.0   
   print('\n\n-------------\n', filename_for_initial_condition, '\ns1, s2', s1, s2)
   ##
   starttime = time.time()
   ##
   p1_found_thisrun, p2_found_thisrun, discoverytime_p1, discoverytime_p2, fixation_thisrun = 0, 0, 0, 0, 0
   P_g_new = np.zeros((N,9), dtype='int16') #population genotypes
   P_p_current = np.zeros(N, dtype='uint32') #array for phenotypes
   P_fitness_current = np.zeros(N, dtype='float32') #array for fitnesses
   #initial conditions
   P_g_current = deepcopy(np.load(filename_for_initial_condition))
   assert len([individual for individual in range(N) if phenotypes[tuple(P_g_current[individual,:])] == p0]) > 0.5 #most initial genotypes are on NC
   print('fraction of initial genotypes on NC:',len([individual for individual in range(N) if phenotypes[tuple(P_g_current[individual,:])] == p0])/float(N) )
   time_initialisation = time.time() - starttime
   P_p_current = np.tile(p0, (N))
   P_fitness_current = np.tile(fitness(p0), (N))
   np.random.seed(random.randint(0, 10**5))
   ########start simulation
   for t in range(0, int(Tmax)):
      #choose next generation
      rows = np.random.choice(np.arange(N), p=np.divide(P_fitness_current[:], np.sum(P_fitness_current[:])), size=N, replace=True)
      #if p1 in P_p_current or p2 in P_p_current: #check that selection works
      #   print(np.mean([f for f in P_fitness_current if f > 0]), np.mean([P_fitness_current[r] for r in rows]))
      for individual in range(N):
         for genotype_index in range(9):
            P_g_new[individual, genotype_index] = P_g_current[rows[individual], genotype_index] #fill in genes for next generation
            #mutate with p=mu
            mutate = random.uniform(0, 1)
            if mutate < mu:
               if P_g_new[individual,genotype_index] == phenotypes.shape[genotype_index] - 1: 
                  P_g_new[individual,genotype_index] =  int(P_g_new[individual,genotype_index]-1)
               elif P_g_new[individual,genotype_index] == 0: 
                  P_g_new[individual,genotype_index] = int(P_g_new[individual,genotype_index]+1)
               else:
                  P_g_new[individual,genotype_index] = int(random.choice([P_g_new[individual,genotype_index]-1, P_g_new[individual,genotype_index]+1]))
         #fill in phenotype and fitness
         P_p_current[individual] = phenotypes[tuple(P_g_new[individual,:])]
         P_fitness_current[individual] = fitness(P_p_current[individual])
      P_g_current[:,:] = deepcopy(P_g_new[:,:])
      if np.count_nonzero(P_fitness_current)==0:
         raise RuntimeError( 'Error: Only zero-fitness individuals left in the population.')
      number_p1_in_pop, number_p2_in_pop = (P_p_current == p1).sum(), (P_p_current == p2).sum()
      if p1_found_thisrun == 0 and number_p1_in_pop > 0.5:
         p1_found_thisrun, discoverytime_p1 = 1, t
      if p2_found_thisrun == 0 and number_p2_in_pop > 0.5:
         p2_found_thisrun, discoverytime_p2 = 1, t
      #check for fixation of p1 or p2
      if number_p1_in_pop > 0.7*N:
         fixation_thisrun = 1
         print( 'fixation of p1')
         break
      elif number_p2_in_pop > 0.7*N:
         fixation_thisrun = 2
         print( 'fixation of p2')
         break
      #if t%50000==0 and t!=0:
      #   print('Time Step: '+str(t))
      #   print( str(float(t)/Tmax*100)+'% finished')
   time_total = time.time() - starttime
   print('finished ', filename_for_initial_condition.split('_')[-1][:-2], 'th run in ', (time_total)/60.0**2, 'hours, out of which initialisation took', time_initialisation * 100.0 /time_total, '%')
   return fixation_thisrun, p1_found_thisrun, p2_found_thisrun, discoverytime_p1, discoverytime_p2


def initialise_population_on_NC(index_and_startgene, N, mu, p0, factor_initialisation, phenotypes): 
   def fitness_initialisation(phenotype):
      if int(phenotype) == int(p0):
         return 1.0
      else:
         return 0.0   
   print(index_and_startgene[0], 'th repetition starting at genotype - start initialisation ', index_and_startgene[1])
   P_g_new = np.zeros((N,9), dtype='int16') #population genotypes
   P_g_current = np.zeros((N,9), dtype='int16') #population genotypes
   P_p_current = np.zeros(N, dtype='uint32') #array for phenotypes
   P_fitness_current = np.zeros(N, dtype='float32') #array for fitnesses
   #initial conditions
   if phenotypes[index_and_startgene[1]] != p0:
      raise RuntimeError('Something went wrong with the start genotype.')
   #use initial conditions
   P_g_current = np.tile(index_and_startgene[1], (N,1)) #use initial condition for population
   P_p_current = np.tile(p0, (N))
   P_fitness_current = np.tile(fitness_initialisation(p0), (N))
   ########start simulation
   tmax = factor_initialisation * int(N)
   for t in range(0, tmax):
      #choose next generation
      rows = random.choices(np.arange(N), weights=np.divide(P_fitness_current[:],float(np.sum(P_fitness_current[:]))), k=N) #select N rows that get chosen for the next generation
      for individual in range(N):
         for genotype_index in range(9):
            P_g_new[individual, genotype_index] = P_g_current[rows[individual], genotype_index] #fill in genes for next generation
            #mutate with p=mu
            mutate = random.uniform(0, 1)
            if mutate < mu and t != tmax - 1: #do not mutate at last step
               if P_g_new[individual,genotype_index] == phenotypes.shape[genotype_index] - 1: 
                  P_g_new[individual,genotype_index] =  int(P_g_new[individual,genotype_index]-1)
               elif P_g_new[individual,genotype_index] == 0: 
                  P_g_new[individual,genotype_index] = int(P_g_new[individual,genotype_index]+1)
               else:
                  P_g_new[individual,genotype_index] = int(random.choice([P_g_new[individual,genotype_index]-1, P_g_new[individual,genotype_index]+1]))
         #fill in phenotype and fitness
         P_p_current[individual] = phenotypes[tuple(P_g_new[individual,:])]
         P_fitness_current[individual] = fitness_initialisation(P_p_current[individual])
      P_g_current[:,:] = deepcopy(P_g_new[:,:])
      if np.count_nonzero(P_fitness_current)==0:
         raise RuntimeError( 'Error: Only zero-fitness individuals left in the population.')
   print('finished initialisation for', index_and_startgene[0], 'th run for ', index_and_startgene[1])
   return P_g_current
