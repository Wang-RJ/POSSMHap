# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

class rbMutationBlock:
    def __init__(self, size, mutChrom, mutPos, phaseDF = None):
        self.size = size
        self.informativeSites = {}
        
        self.mutSide  = None
        self.mutLocus = (mutChrom, mutPos)

        if phaseDF is not None:
            self.importDF(phaseDF)
            
    def importDF(self, phaseDF):
        mutChrom, mutPos = self.mutLocus
        dist = self.size // 2
        
        distanceMatch = phaseDF['CHROM'] == mutChrom
        positionMatch = phaseDF['POS'].between(mutPos - dist, mutPos + dist)
        region = phaseDF[distanceMatch & positionMatch]

        # get genotype and blockID of the mutation
        mutIdx = region.index[region['POS'] == mutPos]
        self.mutSide, mutBlock = getMutation(region, mutIdx)
        
        # get informative sites for phasing, exclude the mutation itself
        genotypes, phaseArray = splitGT(region.drop(mutIdx))

        # no informative sites if no mutation phase, no genotypes, or no child phases
        if(mutBlock is None
           or genotypes is None
           or phaseArray['Child'].isna().all()):
            return

        # include all positions in the region with phase information
        contained = findContainedPhase(mutBlock, phaseArray)
        genotypes = genotypes.iloc[contained]

        # dictionary of informative site arrays (n x 2) for Mother, Father, and Child
        self.informativeSites = { member : makeInformative(genotypes[member])
                                  for member in genotypes }

def calculate_rbPhase(mutBlock, minSupport = 1):
    if not mutBlock.informativeSites:
        return ''
        
    def hamDist(x, y):
        return np.sum(x != y)

    # There are 2 haplotypes that a child inherits and we want to find the
    # configuration of transmitted haplotypes that minimizes the distance between them.
    # There are 8 pairwise distances between 3 different pairs of haplotypes.
    # There are also 8 possible configurations from transmission, e.g. no uniparental disomy
    #
    # Let C_0 = [c0, c1] be a vector of the child haplotypes and
    # M = [[m0, m0, m1, m1], [p0, p1, p0, p1]] be a matrix where columns represent
    # configurations of maternal and paternal haplotype transmission
    #
    # C_0 x M, where the hamming distance replaces multiplication between pairs
    # of elements in the matrix operation, gives a vector of distances for all possible
    # configurations where c0 is maternally inherited and c1 is paternally inherited
    # Let C_1 = [c1, c0], then C_1 x M gives the vector of distances for all possible
    # configurations where c0 is paternally inherited and c1 is maternally inherited
    
    # C_0 x M and C_1 x M are calculated below as configs0 and configs1:
    informative = mutBlock.informativeSites
    
    configs0 = [hamDist(informative['Child'][0], informative['Mother'][i]) +
                hamDist(informative['Child'][1], informative['Father'][j])
               for i in range(2) for j in range(2)]
    configs1 = [hamDist(informative['Child'][1], informative['Mother'][i]) +
                hamDist(informative['Child'][0], informative['Father'][j])
               for i in range(2) for j in range(2)]
    
    # matching requires difference between two smallest configs to be non-zero
    configs = configs0 + configs1
    if sorted(configs)[1] - sorted(configs)[0] < minSupport:
        return ''
    
    # matching config must have distance 0
    if min(configs0) == 0:
        phase = 0
    elif min(configs1) == 0:
        phase = 1
    else:
        return ''

    # flip phase for the mutation if it's on c1 instead of c0
    phase = (1-phase) if mutBlock.mutConfig else phase
    
    phaseVals = ['Maternal', 'Paternal']
    return phaseVals[phase]

# get genotype and block ID for mutation in child from vcf DF
def getMutation(df, idx):
    if len(df) < 1:
        return None, None
    
    # a phased genotype entry in a vcf has two elements split by a colon   
    entry = df.loc[idx, 'Child'].item().split(':')

    if len(entry) > 1:
        geno, block = entry[0], entry[1]
    else:
        geno, block = entry[0], None

    if geno == '1|0':
        cfg = 0
    elif geno == '0|1':
        cfg = 1
    else:
        cfg = None
        
    return cfg, block

def splitGT(df):
    # utility to split by colon across the array
    # also clean up phases by replacing "." with None
    def splitColumns(vcfGT):
        genos, phases = pd.DataFrame(), pd.DataFrame()
        
        for individual in vcfGT:
            newCols = vcfGT[individual].str.split(':', expand = True)
                
            genos[individual]  = newCols[0]
            phases[individual] = newCols[1] if len(newCols.columns) > 1 else None
        
        return genos, phases.replace('.', None)
    
    # split the df and reset the index
    if len(df) < 1:
        return None, None
    else:
        return splitColumns(df.reset_index(drop = True) \
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
        contained = (minIdx,)

    return list(contained)

# turn an individual's genotypes into an 2-column array of informative sites
def makeInformative(gtCol):
    # Single heterozygote only possible when informative homozygous site exists in
    # other parent, this parent can then be given an arbitrary phase
    if (gtCol == '0/1').sum() < 2:
        gtCol = gtCol.replace('0/1', '0|1')
            
    # Count homozygote genotypes as phase informative
    gtCol = gtCol.replace({
        '1/1': '1|1',
        '0/0': '0|0',
        })
                
    return gtCol.str.split("\\|", expand = True)
