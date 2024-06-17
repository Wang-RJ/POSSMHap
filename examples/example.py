# -*- coding: utf-8 -*-

import rbPhase

# Import individual read based phases from Whatshap,
# Genotypes of the trio must be in order: Mother, Father, Child
phasedVCF = pd.read_csv("phasedTrio.vcf", comment = '#', sep = "\t",
                         names = ['CHROM', 'POS', 'Mother', 'Father', 'Child'])

# Mutation positions in bed format
mutations = pd.read_csv("mutations.bed", sep = "\t")

# Form an rbMutationBlock by passing the following parameters:
#   size     - size of the mutation block
#   mutChrom - chromosome of mutation
#   mutPos   - position of mutation
#   phaseDF  - the phased vcf imported as a pandas dataframe

mublock = rbPhase.rbMutationBlock(size = 10000, 
                                  mutations.loc[2, 'CHROM'],
                                  mutations.loc[2, 'POS'],
                                  phasedVCF)

# Call calculate_rbphase with the mutation block, returns: 'Mother', 'Father', or ''
muphase = rbPhase.calculate_rbPhase(mublock)

# Calculate phase for every mutation using a list of block objects
phase_blocks = [rbPhase.rbMutationBlock(10000, c, p, phaseOutput) for c, p in
                 zip(mutationPositions["CHROM"], mutationPositions["POS"])]
mu_phases = [rbPhase.calculate_rbPhase(block) for block in phase_objects]

# Print phases wih the positions of the mutations
print([(mutations.iloc[i].tolist(), x) for i,x in enumerate(phase_list) if x != ''])
