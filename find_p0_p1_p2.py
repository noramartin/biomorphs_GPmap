import numpy as np
from functions_for_GP_analysis.neutral_component import NC_rho_c_pq_biomorphs, get_NC_genotypes
import sys
from functions_for_GP_analysis.GPproperties import rho_phi_pq_biomorphs, nonzero_phi_pq_biomorphs
import pandas as pd
from os.path import isfile

#####################
## need to find three phenotypes p0, p1, p2
##### p0 should be relatively common (but small enough to draw initial NC)
##### from there, find pairs of phenotypes that have very different c_pq
###### both must have a connection to p0, but not to each other


   
#######################################################################################################################################
genemax = int(sys.argv[1])
g9min = int(sys.argv[2])
g9max = int(sys.argv[3])
resolution = int(sys.argv[4])
threshold = int(sys.argv[5])



string_for_saving_plot=str(genemax)+'_'+str(g9min)+'_'+str(g9max) +'_'+str(resolution) + '_'+str(threshold)
phenotypes=np.load('./biomorphs_map_pixels/phenotypes'+string_for_saving_plot+'.npy')
#######################################################################################################################################
print('collect data')
#######################################################################################################################################
df_GPmap_properties = pd.read_csv('./biomorphs_map_pixels/GPmap_properties'+string_for_saving_plot+'.csv')
ph_vs_N = {ph: N for ph, N in zip(df_GPmap_properties['phenotype int'].tolist(), df_GPmap_properties['neutral set size'].tolist())} 
ph_vs_complexity_estimate = {ph: c for ph, c in zip(df_GPmap_properties['phenotype int'].tolist(), df_GPmap_properties['array complexity BDM'].tolist())} 
phenotypes_sorted_by_freq_rank = sorted([ph for ph in ph_vs_N.keys()], key=ph_vs_N.get, reverse=True)

#######################################################################################################################################
print('start analysis')
#######################################################################################################################################
candidate_trios = {} #save trio as tuple with info on phi_pq1 and phi_pq2 in list as value
enough_candidates_found=False
filename_results = './arrival_frequent_data/p0_p1_p2_candidates_vs_NCstartgene_N0_N1_N2_cpq1_cpq2'+string_for_saving_plot+'.npy'
if not isfile(filename_results):
   for p0 in phenotypes_sorted_by_freq_rank:
      if enough_candidates_found or ph_vs_N[p0] > 2 * 10**4: #these would be too much to draw
         continue
      print( 'neutral set size: ', ph_vs_N[p0])
      #find one gene with that phenotype
      startgene_ind = tuple([g for g in np.argwhere(phenotypes==p0)[np.random.randint(0, ph_vs_N[p0]-1)]])
      c_pq, rho = NC_rho_c_pq_biomorphs(phenotypes, startgene_ind)
      V = get_NC_genotypes(startgene_ind, phenotypes, max_index_by_site = np.array([phenotypes.shape[pos] - 1 for pos in range(len(phenotypes.shape))]))
      assert len(V) <= ph_vs_N[p0]
      if len(V)<10**5: #otherwise not possible to draw NC as network (if too many nodes)
         for p1 in c_pq:
            for p2 in c_pq:
               KC0, KC1, KC2 = [ph_vs_complexity_estimate[p] for p in [p0, p1, p2]]
               if p1!=p0 and p2!=p0 and ph_vs_N[p1]<ph_vs_N[p0] and ph_vs_N[p2]<ph_vs_N[p0] and p1 < p2  and 1.5 < KC1/KC0 and 1.5 < KC2/KC0:#want to transitions to complex phenotypes
                  bias = max(c_pq[p1]/c_pq[p2], c_pq[p2]/c_pq[p1])
                  if 10 <= bias <= 100 and (c_pq[p1]>0 and c_pq[p2]>0) and 0.01 > max(c_pq[p1], c_pq[p2]) > 0.0005:  #intermediate bias (for computational tractabiility) and transitions not too frequent (if cpq is close to 1, it is trivial)
                     #find phi_pq of p1
                     if c_pq[p2] > c_pq[p1]:
                        p1, p2 = p2, p1
                     phi_pq_p_nonzero2 = nonzero_phi_pq_biomorphs(phenotypes, p2) #this is one way of checking that there is no connection between p1 and p2 - this requirement is actually stricter than it needs to be (not any connection between p1 and p2 can create a bridge for this example), but easy to check
                     if p1 not in phi_pq_p_nonzero2:
                        candidate_trios[(p0, p1, p2)]=[startgene_ind, ph_vs_N[p0], ph_vs_N[p1], ph_vs_N[p2], c_pq[p1], c_pq[p2]]
                        print( 'new candidate: ', (p0, p1, p2), [startgene_ind, ph_vs_N[p0], ph_vs_N[p1], ph_vs_N[p2], c_pq[p1], c_pq[p2]], '\n\n')
                        np.save('./arrival_frequent_data/p0_p1_p2_candidates_vs_NCstartgene_N0_N1_N2_cpq1_cpq2'+string_for_saving_plot+'.npy', candidate_trios)
                        continue
                     del phi_pq_p_nonzero2
                     if len(set([p0 for (p0, p1, p2) in candidate_trios.keys()])) > 5:
                        enough_candidates_found=True
         del V, c_pq
         if enough_candidates_found:
            break
   np.save(filename_results, candidate_trios)
#######################

#######################################################################################################################################
print('print data')
#######################################################################################################################################
candidate_trios=np.load(filename_results, allow_pickle=True).item()
for p0, p1, p2 in candidate_trios:
   startgene_ind, N0, N1, N2, c_pq1, c_pq2=candidate_trios[(p0, p1, p2)]
   bias = max(c_pq1/c_pq2, c_pq2/c_pq1)
   rarer_is_more_complex = (1.3 <  ph_vs_complexity_estimate[p2]/ph_vs_complexity_estimate[p1])
   new_phenotypes_are_more_complex = (1.5 <  ph_vs_complexity_estimate[p2]/ph_vs_complexity_estimate[p0] and 1.5 <  ph_vs_complexity_estimate[p1]/ph_vs_complexity_estimate[p0])
   if rarer_is_more_complex and new_phenotypes_are_more_complex and bias > 9.5:
         c_pq, rho = NC_rho_c_pq_biomorphs(phenotypes, startgene_ind)
         V = get_NC_genotypes(startgene_ind, phenotypes, max_index_by_site = np.array([phenotypes.shape[pos] - 1 for pos in range(len(phenotypes.shape))]))
         assert phenotypes[tuple(startgene_ind)] == p0
         print( 'startgene (in indices): ', startgene_ind)
         print( 'p0='+str(p0))
         print( 'p1='+str(p1))
         print( 'p2='+str(p2))
         print( 'c_01='+str(c_pq1))
         print( 'c_02='+str(c_pq2))
         print( 'NC size='+str(len(V)))
         print( 'bias: '+str(bias))
         print( 'KC1: '+str(ph_vs_complexity_estimate[p1]))
         print( 'KC2: '+str(ph_vs_complexity_estimate[p2]), '\n\n\n\n')


