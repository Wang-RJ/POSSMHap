import pandas as pd
from pouch import *

# Argparse
import argparse
parser = argparse.ArgumentParser(description="Script to run POUCH")
parser.add_argument("vcf", help="Full path of the vcf file that were formatted by Whatshap. Default separator is tab. e.g., 'phased.vcf'")
parser.add_argument("trio_order", help="Order of the trio in the vcf file. e.g., 'Mother,Father,Child'")
parser.add_argument("mutation_file", help="Full path of the mutation file in bed format. Default separator is tab. e.g., 'mutations.bed'")
parser.add_argument("output", help="Full path of the output file. e.g., 'output.txt'")

args = parser.parse_args()

# Assign arguments to variables
vcf = args.vcf
trio_order = args.trio_order.split(",")
mutation_file = args.mutation_file
output = args.output

## Script to run pouch
# Import individual read-based phases from Whatshap
phased_vcf = pd.read_csv(vcf, sep="\t", comment="#", header=None, names=["CHROM", "POS"] + trio_order)

# Mutation positions in bed format
mutations = pd.read_csv(mutation_file, sep="\t")

# Calculate phaseBlock for every mutation
phase_blocks = [
        RbMutationBlock(10000, chrom, pos, phased_vcf)
        for chrom, pos in zip(mutations["CHROM"], mutations["POS"])
    ]

# Extract phases
mu_phases = [ 
        (block.mut_locus, block.phase, block.phase_method) 
        for block in phase_blocks if block.phase in ["Maternal", "Paternal"]]

not_phased = [
        (block.mut_locus, block.phase, block.phase_method)
        for block in phase_blocks if block.phase not in ["Maternal", "Paternal"]
    ]


# Write output in a file
with open(output, "w") as f:
    f.write("Mutation\tParentOfOrigin\tMethod\n")
    for mut, phase, method in mu_phases:
        f.write(f"{mut} {phase} {method}\n")
    for mut, phase, method in not_phased:
        f.write(f"{mut} {phase} {method}\n")