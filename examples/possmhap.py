# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

class rbMutationBlock:
    def __init__(self, size, mutChrom, mutPos, phaseDF = None):
        self.size = size
        self.informativeSites = {}
        
        self.mutCfg  = None
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
        self.mutCfg, mutBlock = getMutation(region, mutIdx)
        
        # get informative sites for phasing, exclude the mutation itself
        genotypes, phaseArray = splitGT(region.drop(mutIdx))

        # for debug
        self.genotypes, self.phaseArray = genotypes, phaseArray

        # no informative sites if no mutation phase, no genotypes, or no child phases
        if(mutBlock is None
           or genotypes is None
           or phaseArray['Child'].isna().all()):
            return

        # include all positions in the region with phase information
        contained = findContainedPhase(mutBlock, phaseArray)
        genotypes = genotypes.iloc[contained]

        # informative sites are a dict of arrays (n x 2) for keys: Mother, Father, Child
        self.informativeSites = makeInformative(genotypes)

def calculate_rbPhase(mutBlock, minSupport = 1, maxDistance = 1):
    if not mutBlock.informativeSites:
        return ''
        
    def hamDist(x, y):
        return np.sum(x != y)
    
    # C_0 * M and C_1 * M are calculated below as configs(0) and configs(1):
    informative = mutBlock.informativeSites
    
    def configs(k):
        return [hamDist(informative['Child'][k],   informative['Mother'][i]) +
                hamDist(informative['Child'][1-k], informative['Father'][j])
                for i in range(2) for j in range(2)]
    
    # matching requires difference between configs to be non-zero
    if abs(min(configs(1)) - min(configs(0))) < minSupport:
        return ''
    
    # matching config must not exceed maxDistance
    if min(configs(0)) <= maxDistance:
        phase = 0
    elif min(configs(1)) <= maxDistance:
        phase = 1
    else:
        return ''

    # if the mutation is on c0, phase variable is consistent with the min-distance
    # config, otherwise flip it
    phase = (1-phase) if mutBlock.mutCfg else phase
    
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

# Turn each individual's genotypes into an 2-column array of informative sites
def makeInformative(genotypes):
    # Give heterozygotes in parents a phase if it's the only transmittable genotype,
    # This allows us to phase e.g.,
    #           Mother    Father    Child
    #           0/0       0/1         0|1

    if (genotypes[['Mother','Father']] == '0/1').values.sum() == 1:
        genotypes = genotypes.replace('0/1', '0|1')
        
    genotypes = genotypes.replace({
        '1/1': '1|1',
        '0/0': '0|0',
        '0/1': None
        })
    # Count homozygote genotypes as phase informative, drop unphased heterozygotes
    genotypes = genotypes.dropna()
    
    if any((len(genotypes[member].index) < 1 for member in genotypes)):
        return {}
    else:
        return { member : genotypes[member].str.split("\\|", expand = True)
                 for member in genotypes }