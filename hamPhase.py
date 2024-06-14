# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

class rbPhaseBlock:
    def __init__(self, phasedDF, muChrom, muPos, dist):
        self.phase = None
        
        distanceMatch = phasedDF['CHROM'] == muChrom
        positionMatch = phasedDF['POS'].between(muPos - dist, muPos + dist)
        region = phasedDF[distanceMatch & positionMatch]

        # get genotype and blockID of the mutation
        muIdx = region.index[region['POS'] == muPos]
        self.muIndicator, muBlock = getMutation(region, muIdx)
        
        # get informative sites for phasing, exclude the mutation itself
        genotypes, phaseArray = splitGenotypeData(region.drop(muIdx))

        # cannot phase if no mutation phase, no genotypes, or no child phases
        if(muBlock is None
           or genotypes is None
           or phaseArray['Child'].isna().all()):
            self.phase = np.nan
            return

        # include all positions in the region with phase information
        contained   = findContainedPhase(muBlock, phaseArray)
        informative = genotypes.iloc[contained]

        self.informativeBlocks = { member : splitPhasedInformative(informative[member])
                                   for member in informative }

    def getPhase(self, minSupport = 1):
        if self.phase is not None:
            return self.phase
        else:
            self.phase = self.calculatePhase(minSupport)
            return self.phase

    def calculatePhase(self, minSupport):
        def flip(dummy):
            return 1-dummy
        
        def hamDist(x, y):
            return np.sum(x != y)

        # broadcast hamming distance calculation across combinations        
        def distanceMat(block1, block2):
            return np.array(
                [[hamDist(block1[i], block2[j]) for j in range(2)]
                 for i in range(2)])
        
        # calculate 4 pairwise distances from each parent-child comparison
        infoB = self.informativeBlocks        
        matD = distanceMat(infoB['Mother'], infoB['Child'])        
        patD = distanceMat(infoB['Father'], infoB['Child'])
        
        # find the minimum distance between parent and child's Left and Right haploblocks
        # Left:  x|.
        # Right: .|x
        nDiffLeft  = abs(min(matD[0,]) - min(patD[0,]))
        nDiffRight = abs(min(matD[1,]) - min(patD[1,]))
        
        # someone must meet the minimum number of informative sites to support a haploblock
        if nDiffLeft < minSupport and nDiffRight < minSupport:
            return np.nan
        
        # Left:        x|.             Right:  .|x
        # muInd == 0:  1|0        muInd == 1:  0|1
        #
        # mu indicator matches parents on Left, flip on a Right match
        if nDiffLeft > nDiffRight:
            indicator = self.muIndicator
        else:
            indicator = flip(self.muIndicator)
        
        phaseVals = ['Mother', 'Father']
        # indicator == 0 defaults to Mother, flip on match to Father
        # matching haploblock must have 0 distance, otherwise return nan
        if min(matD[indicator,].flat) == 0:
            return phaseVals[indicator]
        elif min(patD[indicator,].flat) == 0:
            return phaseVals[flip(indicator)]
        else:
            return np.nan

# get genotype and block ID for mutation in child from vcf DF
def getMutation(df, idx):
    # a phased genotype entry in a vcf has two elements split by a colon   
    entry = df.loc[idx, 'Child'] \
            .item() \
            .split(':')

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

def splitGenotypeData(df):
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
        }, regex=True)
                
    genoList  = genoColumn.str.split("|")
    flattened = [item for sublist in genoList for item in sublist]
                
    return np.array(flattened).reshape(-1, 2)
