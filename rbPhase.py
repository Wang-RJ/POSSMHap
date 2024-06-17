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

        # dictionary of informative sites for Mother, Father, and Child
        self.informativeSites = { member : makeInformative(genotypes[member])
                                  for member in genotypes }

def calculate_rbPhase(mutBlock, minSupport = 1):
    if not mutBlock.informativeSites:
        return ''
    
    def flip(dummy):
        return 1-dummy
        
    def hamDist(x, y):
        return np.sum(x != y)

    # broadcast hamming distance calculation across combinations
    def distanceMat(block1, block2):
        return np.array(
            [[hamDist(block1[i], block2[j]) for i in range(2)]
             for j in range(2)])
        
    # calculate 4 pairwise distances from each parent-child comparison
    # 2x2 matrices, rows correspond to the Left and Right haplotypes in the Child
    # Left:  x|.
    # Right: .|x
    informative = mutBlock.informativeSites
    maternalD = distanceMat(informative['Mother'], informative['Child'])        
    paternalD = distanceMat(informative['Father'], informative['Child'])
        
    # find the difference in the minimum distance between each
    # parent and child's Left, then Right haploblocks
    nDiffLeft  = abs(min(maternalD[0,]) - min(paternalD[0,]))
    nDiffRight = abs(min(maternalD[1,]) - min(paternalD[1,]))
        
    # someone must meet the minimum number of informative sites to support a haploblock
    if nDiffLeft < minSupport and nDiffRight < minSupport:
        return ''
        
    # Left:          x|.               Right:  .|x
    # mutSide == 0:  1|0        mutSide == 1:  0|1
    #
    # mutSide == 0 defaults to Left match, flip on a Right match
    if nDiffLeft > nDiffRight:
        indicator = mutBlock.mutSide
    else:
        indicator = flip(mutBlock.mutSide)

    phaseVals = ['Mother', 'Father']
    # indicator == 0 defaults to Mother, flip on match to Father
    # matching haploblock must have 0 distance, otherwise return None        
    if min(maternalD[indicator,].flat) == 0:
        return phaseVals[indicator]
    elif min(paternalD[indicator,].flat) == 0:
        return phaseVals[flip(indicator)]
    else:
        return None

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
        ind = 0
    elif geno == '0|1':
        ind = 1
    else:
        ind = None
        
    return ind, block

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
