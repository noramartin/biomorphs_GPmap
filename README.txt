This folder contains the code required for the paper "Bias in the arrival of variation can dominate over natural selection in Richard Dawkins' biomorphs" (Nora S. Martin, Chico Q. Camargo, Ard A. Louis)

This code uses Python 3.7, and various standard Python packages (matplotlib, pandas etc.).

The main datafile used is a representation of the computational GP map: this is saved as an array, where the array indices represent the genotype and the integer value of the array represents the phenotype. To convert from a genotype tuple to array indices, simply add a constant, such that the lowest allowed value corresponds to '0' in the array index (for example, if a vector site can take any integer between -3 and 3, then a -3 at the genotype site corresponds to a 0 in the array index). Each phenotype is a 2D binary array that can be converted to a tuple of two integers, as specified in the function compress_binary_string(). Then we allocate one unique integer to each tuple and save this mapping as a dictionary.


The code uses the following references:
- Some functions for the GP analysis and plots are adapted from my previous papers (Martin and Ahnert J. Roy Soc. Interface 2021 & 200, Martin and Ahnert, EPL 2022).
- The biomorphs model itself is implemented following the description in Dawkins R. The evolution of evolvability. In: Langton CG, editor. Artificial life : the proceedings of an Interdisciplinary Workshop on the Synthesis and Simulation of Living Systems, held September, 1987 in Los Alamos, New Mexico. Proceedings volume in the Santa Fe Institute studies in the sciences of complexity ; v. 6. Redwood City, Calif ; Wokingham: Addison-Wesley; 1989.
- The method for identifying NCs in large neutral sets efficiently follows Grüner, W., Giegerich, R., Strothmann, D., Reidys, C., Weber, J., Hofacker, I.L., Stadler, P.F. and Schuster, P., 1996. Analysis of RNA sequence structure maps by exhaustive enumeration II. Structures of neutral networks and shape space covering. Monatshefte für Chemie/Chemical Monthly, 127(4), pp.375-389.
- Dijkstra's algorithm is implemented following the Wikipedia article https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm .
- The code relies on a function called calc_KC that implements the LZ-algorithm with some modifications, in order to estimate descriptional complexities. This function is from Dingle et al. 2018.
- The code also relies on the block decomposition method for complexity estimates (Zenil et al. 2018). For this, the module PyBDM was used.
- All other sources for the models and assumptions used in the code are detailed in the paper.

The scripts in the function folder contain the following content:
- biomorph_functions.py general functions for plotting biomorphs.
- genotype_to_raster.py implements the computational coarse-graining.
- neutral_component.py extracts a connected neutral component.
- GPproperties.py contains general scripts for computing GP map properties like robustness etc.- fixation.py runs the adaptive simulations in the three-peaked scenario.

The scripts in the main folder contain the following content:
- branches_data_parallel.py runs the coarse-grained phenotype construction, concatenates them to a 1D string and saves them as a text file (some genotypes are not included since their phenotype can be deduced from the axial symmetry of the biomorph system).
- phenotypes_from_text_files_parallel.py takes the 1D binary string and stores this data in a numpy array: in this array, each phenotype is denoted by a unique integer. A mapping between these integers and the binary arrays is saved as a dictionary (here the binary arrays are converted to base-10 numbers for memory reasons). The phenotype of genotype g can ba accessed at index (g1 + 3, g2 + 3, g3 + 3, g4 + 3, g5 + 3, g6 + 3, g7 + 3, g8 + 3, g9 + 1). These constant offsets are required to prevent negative indices.
- array_complexity.py and array_complexity_BDM.py compute the complexity of each phenotype for the LZ-compressor from Dingle et al. 2018 and for the block decomposition method (BDM) by Zenil et al. (2018) respectively.
- plot_GPproperties.py extracts and plots the GP map properties, such as phenotypic bias, robustness, evolvability etc.
- random_walk_barchart.py is for the evolutionary simulation on a flat fitness landscape.
- run_fixation_function_2D_plot.py find_p0_p1_p2.py fixation_plots_2Dplot.py construct_graph_p0_p1_p2.py are for the two-peaked simulations (including finding a suitable trip of phenotypes).
- definition_special_genotypes_phenotypes.py is a parameter file that holds information on the three-peaked landscape and the beetle phenotype.
- beetle_from_dot_genotype_path.py and beetle_from_dot_phenotype_path.py are for the shortest genotype/phenotype path in the search for the insect-shaped phenotype.


 