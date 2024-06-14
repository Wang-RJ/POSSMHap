# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

phaseOutput = pd.read_csv("output.trio1.vcf", sep = "\t")
phaseOutput.columns = ['CHROM', 'POS', 'Mother', 'Father', 'Child']

mutations = pd.read_csv("dnc_table.txt", sep = "\t")

phase = phaseMutation(mutationPositions.iloc[9,0], mutationPositions.iloc[9,1], 1500)

class phaseBlock:   
    def __init__(self, phaseVCF, muChrom, muPos, dist):    
        distanceMatch = phaseVCF['CHROM'] == muChrom
        positionMatch = phaseVCF['POS'].between(muPos - dist, muPos + dist)

        region = phaseVCF[distanceMatch & positionMatch]

        # find indicator for genotype and blockID for mutation
        muIdx = region.index[region['POS'] == muPos]
        self.muIndicator, self.muBlock = getMutation(region, muIdx)
        
        # get informative sites for phasing, exclude the mutation itself
        genotypes, phaseArray = splitGenotypeData(region.drop(muIdx))

        # include all positions in the region with phase information
        if self.muBlock is not None and self.muBlock != '.':
            contained = findContainedPhase(self.muBlock, phaseArray)
            informative = genotypes[contained]
        else:
            informative = genotypes

        self.informativeBlocks = [splitPhasedInformative(individual)
                                  for individual in informative]

    def calculatePhase(self):
        def hamDist(x, y):
            return np.sum(x != y)

        # function to broadcast hamming distance calculation across combinations        
        def calculateDistances(block1, block2, flag):
            return np.array(
                [[hamDist(block1[i], block2[j]) for j in flag]
                 for i in range(2)])
        
        flag = self.muIndicator
        comp = 0 if flag else 1

        forwardDist    = (calculateDistances(mom, kid, flag),
                          calculateDistances(dad, kid, flag))
        complementDist = (calculateDistances(mom, kid, comp),
                          calculateDistances(dad, kid, comp))
        
        
        mom_minid_a = np.argmin(mk_a)
        dad_minid_a = np.argmin(dk_a)

        mom_minid_b = np.argmin(mk_b)
        dad_minid_b = np.argmin(dk_b)
        
        ndiff_a = abs(mk_a.flat[mom_minid_a] - dk_a.flat[dad_minid_a])
        ndiff_b = abs(mk_b.flat[mom_minid_b] - dk_b.flat[dad_minid_b])
        
        if ndiff_a < 1 and ndiff_b < 1:
            self.phase = np.nan

        if ndiff_a > ndiff_b:
            if mk_a.flat[mom_minid_a] == 0:
                self.phase = "F"
            if dk_a.flat[dad_minid_a] == 0:
                self.phase = "M"
        else:
            if mk_b.flat[mom_minid_b] == 0:
                self.phase = "M"
            if dk_b.flat[dad_minid_b] == 0:
                self.phase = "F"

# get genotype and block ID for mutation in child from vcf DF
def getMutation(df, idx):
    # a phased genotype entry in a vcf has two elements split by a colon
    geno, block = \
        df \
            .loc[idx, 'Child'] \
            .item() \
            .split(':')

    if geno == '1|0':
        ind = 0
    elif geno == '0|1':
        ind = 1
    else:
        raise Exception("Mutation not associated with any informative sites.")
        
    return ind, block

def splitGenotypeData(df):
    # utility to split by colon across the array
    def splitColumns(vcfGT):
        genos, phases = pd.DataFrame(), pd.DataFrame()
        
        for individual in vcfGT:
            newCols = vcfGT[individual].str.split(':', expand = True)
                
            genos[individual]  = newCols[0]
            phases[individual] = newCols[1]
        
        return genos, phases
    
        # split the df and reset the index
    return splitColumns(df \
                            .reset_index(drop = True) \
                            .loc[:, ['Mother', 'Father', 'Child']])

# Find idx of all genotypes with *any* phase information across the mutation block.
# That is, include positions with phase information from either parent if they're
# between two positions marked as phased with the mutation.
def findContainedPhase(muBlock, phaseArr):
    blockMatch = phaseArr.loc[:, 'Child'] == muBlock
    matchIdxs  = phaseArr.index[blockMatch].tolist()
            
    minIdx, maxIdx = min(matchIdxs), max(matchIdxs)
            
    if maxIdx > minIdx:
        notNone   = phaseArr[~phaseArr.isna().all(axis=1)].index
        contained = set(notNone) & set(range(minIdx, maxIdx+1))
    else:
        contained = set(minIdx)

    return list(contained)

# split an individual's informative genotypes into an 2-column array
def splitPhasedInformative(genoColumn):
    # Single heterozygote only possible when informative site exists in
    # other parent, this parent can be given an arbitrary phase
    if (genoColumn == '0/1').sum() < 2:
        genoColumn = genoColumn.replace('0/1', '0|1')
            
        # Count homozygote genotypes as phase informative
    genoColumn = genoColumn.replace({
        '1/1': '1|1',
        '0/0': '0|0',
        '\\.': 'NA|NA'
        }, regex=True)
                
    genoList  = genoColumn.str.split("|")
    flattened = [item for sublist in genoList for item in sublist]
                
    return np.array(flattened).reshape(-1, 2)
