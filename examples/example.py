# -*- coding: utf-8 -*-

# import possmhap
import pouch
import pandas as pd
import possmhap
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

mublock = pouch.rbMutationBlock(10000,
                                  mutations.loc[2, 'CHROM'],
                                  mutations.loc[2, 'POS'],
                                  phasedVCF)

# Call calculate_rbphase with the mutation block, returns: 'Mother', 'Father', or ''
muphase = pouch.calculate_rbPhase(mublock)
print(muphase)

# # Calculate phase for every mutation using a list of block objects
phase_blocks = [possmhap.rbMutationBlock(10000, c, p, phasedVCF) for c, p in
                 zip(mutations["CHROM"], mutations["POS"])]
mu_phases = [possmhap.calculate_rbPhase(block) for block in phase_blocks]

# Print phases wih the positions of the mutations
print([(mutations.iloc[i].tolist(), x) for i,x in enumerate(mu_phases) if x != ''])
