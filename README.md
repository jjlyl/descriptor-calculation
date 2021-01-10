This script is wrote by Python language and is used for molecular descriptor based on multiscale weighted colored graph (MWCG) theory. This molecular descriptor targets pairwise non-covalent interactions of molecules. For a given dataset, we first perform a statistical analysis to identify a set of commonly occurring chemical element types, say C, and for a given molecule or biomolecule in the dataset, there is a subset of N atoms that are members of C. Each atom is labeled both by its element type and its position. The classificaiotn of atoms into chemical element types is a graph coloring, which is important for encoding different types of interactions and give rise to a basis for the collective coarse-grained description of the dataset. The details of calculation of descriptor can be found in the paper ( Nguyen, D. D.; Wei, G. W. AGL-Score: Algebraic Graph Learning Score for Protein-Ligand Binding Scoring, Ranking, Docking, and Screening. J. Chem. Inf. Model. 2019, 59 (7), 3291–3304. https://doi.org/10.1021/acs.jcim.9b00334.)
