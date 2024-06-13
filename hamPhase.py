# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

phaseOutput = pd.read_csv("output.trio1.vcf", sep = "\t")
phaseOutput.columns = ['CHROM', 'POS', 'Mother', 'Father', 'Child']

mutations = pd.read_csv("dnc_table.txt", sep = "\t")

phase = phaseMutation(mutationPositions.iloc[9,0], mutationPositions.iloc[9,1], 1500)

class phaseMutation:
    # utility fun to split genotype and phase information, separated by ":" in vcf
    def splitColumns(vcfGT):
        genotype, phaseID = pd.DataFrame(), pd.DataFrame()
        
        for individual in vcfGT:
            newCols = vcfGT[individual].str.split(':', expand = True)
            genotype[individual] = newCols[0]
            phaseID[individual]  = newCols[1]
        
        return genotype, phaseID
    
    # clean and split an individual's phased genotype into an array
    def splitPhasedGT(genoColumn):
        # Single heterozygote only possible when informative site exists in
        # other parent, this parent can be given an arbitrary phase
        if (genoColumn == '0/1').sum() < 2:
            genoColumn = genoColumn.replace('0/1', '0|1')
        
        genoColumn = genoColumn.replace({
            '1/1': '1|1',
            '0/0': '0|0',
            '\\.': 'NA|NA'
            }, regex=True)
        
        genoList  = genoColumn.str.split("|")
        flattened = [item for sublist in genoList for item in sublist]
        
        return np.array(flattened).reshape(-1, 2)
    
    # return the hamming distance
    def hamDist(x, y):
        return np.sum(x != y)
    
    def __init__(self, muChrom, muPos, dist):
        distanceMatch = phaseOutput['CHROM'] == muChrom
        positionMatch = phaseOutput['POS'].between(muPos - dist, muPos + dist)
        matchedOutput = phaseOutput[distanceMatch & positionMatch]
        
        # get phased genotype and phase block ID at mutation position
        mutationIdx = matchedOutput.index[matchedOutput['POS'] == muPos]
        muGenotype, muBlockID = \
            matchedOutput \
                .loc[mutationIdx, 'Child'] \
                .item() \
                .split(':')

        # drop the mutation entry itself for comparisons and reset the DF index
        genotypes, blockIDs = \
            self.splitColumns(
                matchedOutput \
                    .drop(mutationIdx) \
                    .reset_index(drop = True) \
                    .loc[:, ['Mother', 'Father', 'Child']])

        # find all genotypes with *any* phase information across the mutation block
        # e.g. a position with only paternal phase between two positions with muBlockID
        if muBlockID is not None and muBlockID != '.':
            blockIDMatch    = blockIDs.loc[:, 'Child'] == muBlockID
            matchingIndices = blockIDs.index[blockIDMatch].tolist()
            
            minIdx, maxIdx = min(matchingIndices), max(matchingIndices)
            
            if maxIdx > minIdx:
                notNone   = blockIDs[~blockIDs.isna().all(axis=1)].index
                contained = set(notNone) & set(range(minIdx, maxIdx+1))
            else:
                contained = set(minIdx)
            
            genotypes = genotypes.loc[list(contained)]
        
        momBlock = self.splitPhasedGT(genotypes.loc[:, "Mother"])
        dadBlock = self.splitPhasedGT(genotypes.loc[:, "Father"])
        kidBlock = self.splitPhasedGT(genotypes.loc[:, "Child"])
        
        mu_flag       = 0 if muGenotype == '1|0' else 1
        mu_complement = 0 if mu_flag else 1
        
        def calculate_distances(block1, block2, flag):
            return np.array([[self.hamdist(block1[i], block2[j]) for j in flag] for i in [0, 1]])
        
        mk_a = calculate_distances(momBlock, kidBlock, mu_flag)
        dk_a = calculate_distances(dadBlock, kidBlock, mu_flag)
        mk_b = calculate_distances(momBlock, kidBlock, mu_complement)
        dk_b = calculate_distances(dadBlock, kidBlock, mu_complement)
        
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