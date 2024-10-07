import pandas as pd
import numpy as np
from getMutation import *
from splitGT import *

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

class rbMutationBlock:
    def __init__(self, size, mutChrom, mutPos, phaseDF=None, debug_print=False):
        """
        Initialize an rbMutationBlock object with the mutation position, chromosome, and size of the block.
        Optionally, import phase data from a DataFrame.

        Parameters:
        - size (int): Size of the mutation block.
        - mutChrom (str): Chromosome where the mutation occurs.
        - mutPos (int): Position of the mutation on the chromosome.
        - phaseDF (pd.DataFrame, optional): DataFrame containing phase data.
        - debug_print (bool, optional): If True, prints debug information.
        """
        self.size = size
        self.mutCfg = None
        self.mutLocus = (mutChrom, mutPos)
        self.phase = None         # Phase of the mutation
        self.phaseMethod = ''     # Method used to phase the mutation
        self.haploBlocks = None   # Haploblocks for the child, mother, and father

        # Import phase data if provided
        if phaseDF is not None:
            self.importDF(phaseDF, debug_print)

    def importDF(self, phaseDF, debug_print=False):
        """
        Imports phase data, identifies the region around the mutation,
        and processes genotypes and phases.

        Parameters:
        - phaseDF (pd.DataFrame): DataFrame containing phase data.
        - debug_print (bool, optional): If True, prints debug information.
        """
        # Step 1: Import data and filter the region around the mutation
        mutChrom, mutPos = self.mutLocus
        dist = self.size // 2  # Distance around the DNM to consider

        # Optionally print the DataFrame
        if debug_print:
            print(phaseDF)

        # Filter the region around the mutation position
        distanceMatch = phaseDF['CHROM'] == mutChrom
        positionMatch = phaseDF['POS'].between(mutPos - dist, mutPos + dist)
        region = phaseDF[distanceMatch & positionMatch]  # Region around the De novo candidate

        # Get the index of the De novo candidate
        mutIdx = region.index[region['POS'] == mutPos]

        # Step 2: Get the mutation configuration and haploblocks for the trio
        # mutCfg = '1|0' if the child is 1|0, '0|1' if the child is 0|1 at the DNM position
        self.mutCfg, mutBlockIdDnm = getMutation(region, mutIdx)

        # Get informative sites for phasing, excluding the mutation itself
        genotypes, haploBlocks = splitGT(region)

        # Store genotypes and haploblocks for debugging
        self.genotypes, self.haploBlocks = genotypes, haploBlocks

        # Return if no informative sites or haploblocks
        if mutBlockIdDnm is None or genotypes is None or haploBlocks['Child'].isna().all():
            return

        # Step 3: Find the individual with the longest haploblock containing the DNM position
        longestBlockIndiv, longestBlockId = findIndLongestBlock(haploBlocks, mutBlockIdDnm, mutPos)
        minlongestBlock = haploBlocks[haploBlocks[longestBlockIndiv] == longestBlockId]['POS'].min()
        maxlongestBlock = haploBlocks[haploBlocks[longestBlockIndiv] == longestBlockId]['POS'].max()

        # Find all the haploblocks within the range of the longest haploblock containing the DNM
        HaploBlkChildlst = haploBlocks[haploBlocks["POS"].between(minlongestBlock, maxlongestBlock)]["Child"].unique()
        HaploBlkMotherlst = haploBlocks[haploBlocks["POS"].between(minlongestBlock, maxlongestBlock)]["Mother"].unique()
        HaploBlkFatherlst = haploBlocks[haploBlocks["POS"].between(minlongestBlock, maxlongestBlock)]["Father"].dropna().unique()

        # Convert them into lists of haploblock objects
        allChildBlocks, dnmChildBlock, allFatherBlocks, dnmFatherBlock, allMotherBlocks, dnmMotherBlock = findCvHaploblocks(
            haploBlocks, HaploBlkChildlst, HaploBlkFatherlst, HaploBlkMotherlst, mutBlockIdDnm, mutPos
        )

        # Step 4: Phasing
        # Step 4.1: Phase the DNM block with the longest haploblock in the parents
        phase = phaseDnmBlock(dnmChildBlock, dnmFatherBlock, dnmMotherBlock, haploBlocks, genotypes, mutPos, self.mutCfg)
        if phase is not None:
            self.phase = self.explicitPhase(phase)
            self.phaseMethod = 'Dnm Haploblock'
            return

        # If simple matrix phasing didn't work, try other methods
        print("Simple matrix phasing didn't work. Will try to phase the DNM block with the longest haploblock in the parents")

        if longestBlockIndiv == 'Child':
            # Find haploblocks in parents excluding the DNM block
            Blocklst = [block for block in allFatherBlocks + allMotherBlocks if not block.dnm]

            # Sort the haploblocks by size in decreasing order
            Blocklst.sort(key=lambda x: x.size, reverse=True)

            # Start phasing using the longest haploblock
            while Blocklst:
                longestBlockparent = Blocklst.pop(0)
                commonMinPos = max(longestBlockparent.start, dnmChildBlock.start)
                commonMaxPos = min(longestBlockparent.end, dnmChildBlock.end)
                if commonMaxPos >= commonMinPos:
                    phase = singleParentPhasing(dnmChildBlock, longestBlockparent, genotypes, mutPos, self.mutCfg, debug_print=debug_print)

                if phase is not None:
                    self.phase = self.explicitPhase(phase)
                    self.phaseMethod = 'Phasing using the longest haploblock in the parents'
                    return

        elif longestBlockIndiv in ['Father', 'Mother']:
            # Find haploblocks in the child excluding the DNM block
            Blocklst = [block for block in allChildBlocks if not block.dnm]

            # Sort the haploblocks by size in decreasing order
            Blocklst.sort(key=lambda x: x.size, reverse=True)
            # Get the longest haploblock in the parents
            longestBlock = dnmFatherBlock if longestBlockIndiv == 'Father' else dnmMotherBlock

            # Start phasing using the longest haploblock
            while Blocklst:
                longestBlockchild = Blocklst.pop(0)
                commonMinPos = max(longestBlockchild.start, longestBlock.start)
                commonMaxPos = min(longestBlockchild.end, longestBlock.end)
                if commonMaxPos >= commonMinPos:
                    phase = singleParentPhasing(longestBlockchild, longestBlock, genotypes, mutPos, self.mutCfg, debug_print=debug_print)

                if phase is not None:
                    self.phase = self.explicitPhase(phase)
                    self.phaseMethod = 'Phasing using the longest haploblock in the parents'
                    return

        # Step 4.3: Block chaining method
        print("Phasing using the block chaining method")
        if longestBlockIndiv == 'Child':
            # Sort haploblocks by size in decreasing order for Father and Mother
            sortedFatherBlocks = sorted([block for block in allFatherBlocks if not block.dnm], key=lambda x: x.size, reverse=True)
            sortedMotherBlocks = sorted([block for block in allMotherBlocks if not block.dnm], key=lambda x: x.size, reverse=True)

            for block in sortedFatherBlocks:
                result = self.attemptBlockChaining(dnmFatherBlock, block, haploBlocks, genotypes, longestBlockIndiv, 'Father')
                if result:
                    return

            for block in sortedMotherBlocks:
                result = self.attemptBlockChaining(dnmMotherBlock, block, haploBlocks, genotypes, longestBlockIndiv, 'Mother')
                if result:
                    return

        elif longestBlockIndiv == 'Father':
            sortedChildBlocks = sorted([block for block in allChildBlocks if not block.dnm], key=lambda x: x.size, reverse=True)

            for block in sortedChildBlocks:
                result = self.attemptBlockChaining(dnmChildBlock, block, haploBlocks, genotypes, longestBlockIndiv)
                if result:
                    return

        else:
            sortedChildBlocks = sorted([block for block in allChildBlocks if not block.dnm], key=lambda x: x.size, reverse=True)
            for block in sortedChildBlocks:
                result = self.attemptBlockChaining(dnmChildBlock, block, haploBlocks, genotypes, longestBlockIndiv, debug_print=debug_print)
                if result:
                    return

    def attemptBlockChaining(self, dnmBlock, block, haploBlocks, genotypes, longestBlockIndiv, parentid=None, debug_print=False):
        """
        Helper function to attempt phasing using the block chaining method.

        Parameters:
        - dnmBlock (haplotypeBlock): DNM block of the individual.
        - block (haplotypeBlock): Another block to chain.
        - haploBlocks (pd.DataFrame): DataFrame containing haploblocks.
        - genotypes (pd.DataFrame): DataFrame containing genotypes.
        - longestBlockIndiv (str): Individual with the longest haploblock.
        - parentid (str, optional): Parent identifier ('Father' or 'Mother').

        Returns:
        - bool: True if phasing was successful, False otherwise.
        """
        combined = combineBlocks(dnmBlock, block, haploBlocks, genotypes, mutConfig=self.mutCfg)
        if combined is None:
            return False

        [block0, block1], [block0flip, block1flip], [minBlock0, maxBlock0, minBlock1, maxBlock1] = combined
        c0, c1 = sliceBlocks(genotypes, minBlock0, maxBlock0, minBlock1, maxBlock1, longestBlockIndiv)
        phaseRef = matrixPhasing(block0, block1, c0, c1, parentid=parentid, debug_print=debug_print)
        phaseFlip = matrixPhasing(block0flip, block1flip, c0, c1, parentid=parentid, debug_print=debug_print)

        if (phaseFlip is None and phaseRef is None) or (phaseFlip == phaseRef):
            return False
        if phaseRef is not None:
            self.phase = self.explicitPhase(phaseRef)
            self.phaseMethod = 'Block chaining method'
            return True
        if phaseFlip is not None:
            self.phase = self.explicitPhase(phaseFlip)
            self.phaseMethod = 'Block chaining method'
            return True
        return False

    def explicitPhase(self, phase=None):
        """
        Converts numerical phase value to explicit 'Mother' or 'Father'.

        Parameters:
        - phase (int, optional): Numerical phase value.

        Returns:
        - str: 'Mother' or 'Father' based on the phase value.
        """
        phaseVals = ['Mother', 'Father']
        if phase is not None:
            return phaseVals[phase]
        else:
            return None

# ============================================================================================= #
# Other functions
# ============================================================================================= #

def sliceBlocks(genotypes, minBlock0, maxBlock0, minBlock1, maxBlock1, longestBlockIndiv):
    """
    Slice out the haploblocks that are contained within specified ranges for the individual.

    Parameters:
    - genotypes (pd.DataFrame): DataFrame containing genotypes.
    - minBlock0 (int): Minimum position of block 0.
    - maxBlock0 (int): Maximum position of block 0.
    - minBlock1 (int): Minimum position of block 1.
    - maxBlock1 (int): Maximum position of block 1.
    - longestBlockIndiv (str): Individual identifier ('Child', 'Father', or 'Mother').

    Returns:
    - tuple: Two arrays representing haplotypes b0 and b1.
    """
    Block0 = genotypes[genotypes['POS'].between(minBlock0, maxBlock0)][longestBlockIndiv]
    Block1 = genotypes[genotypes['POS'].between(minBlock1, maxBlock1)][longestBlockIndiv]
    # Concatenate the blocks vertically
    Block = pd.concat([Block0, Block1], axis=0)
    b0, b1 = Block.str.split('|', expand=True)[0].astype(int), Block.str.split('|', expand=True)[1].astype(int)
    return b0, b1

def hamming(x, y):
    """
    Calculate the Hamming distance between two sequences.

    Parameters:
    - x (array-like): First sequence.
    - y (array-like): Second sequence.

    Returns:
    - int: Hamming distance.
    """
    return np.sum(x != y)

def matrixPhasing(c0, c1, b0, b1, parentid=None, debug_print=False):
    """
    Determine the phase based on Hamming distances.

    Parameters:
    - c0, c1 (array-like): Child haplotypes.
    - b0, b1 (array-like): Parent haplotypes.
    - parentid (str, optional): Parent identifier ('Father' or 'Mother').

    Returns:
    - int or None: Phase value or None if indeterminate.
    """
    phase = 1 if parentid == 'Father' else 0
    D_c0c1 = [hamming(c0, b0), hamming(c1, b1), hamming(c0, b1), hamming(c1, b0)]
    if debug_print:
        print(f"Configuration: c0parent0, c1parent1, c0parent1, c1parent0 : {D_c0c1}")
    if D_c0c1.count(0) == 1:
        if D_c0c1.index(0) == 0 or D_c0c1[2] == 0:  # c0 from parentid
            return abs(phase - 1)
        else:
            return phase
    else:
        return None

def combineBlocks(refBlock, flipBlock, haploBlocks, genotypes, mutConfig=None):
    """
    Combine two haploblocks into new blocks for phasing.

    Parameters:
    - refBlock (haplotypeBlock): Reference haploblock.
    - flipBlock (haplotypeBlock): Haploblock to be flipped and combined.
    - haploBlocks (pd.DataFrame): DataFrame containing haploblocks.
    - genotypes (pd.DataFrame): DataFrame containing genotypes.
    - mutConfig (str, optional): Mutation configuration.

    Returns:
    - tuple: Combined blocks and their positions.
    """
    refHaploblock = genotypes[genotypes.POS.between(refBlock.start, refBlock.end)]
    flipHaploblock = genotypes[genotypes.POS.between(flipBlock.start, flipBlock.end)]

    if refBlock.individual == 'Child':
        if mutConfig == '1|0':  # c0 carries the mutation => flip it
            block1ref, block0ref = refHaploblock['Child'].str.split('|', expand=True)[0].astype(int), refHaploblock['Child'].str.split('|', expand=True)[1].astype(int)
        else:  # c1 carries the mutation
            block0ref, block1ref = refHaploblock['Child'].str.split('|', expand=True)[0].astype(int), refHaploblock['Child'].str.split('|', expand=True)[1].astype(int)
    else:
        block0ref, block1ref = refHaploblock[refBlock.individual].str.split('|', expand=True)[0].astype(int), refHaploblock[refBlock.individual].str.split('|', expand=True)[1].astype(int)
        block0flip, block1flip = flipHaploblock[flipBlock.individual].str.split('|', expand=True)[0].astype(int), flipHaploblock[flipBlock.individual].str.split('|', expand=True)[1].astype(int)

    if refBlock.start < flipBlock.start:
        # Combine the refBlock with the flipBlock
        newblock0 = np.concatenate((block0ref, block0flip))
        newblock1 = np.concatenate((block1ref, block1flip))

        # Also flip the combination
        newblock0flip = np.concatenate((block0ref, block1flip))
        newblock1flip = np.concatenate((block1ref, block0flip))

        minblock0 = refBlock.start
        maxblock0 = refBlock.end
        minblock1 = flipBlock.start
        maxblock1 = flipBlock.end
    else:
        newblock0 = np.concatenate((block0flip, block0ref))
        newblock1 = np.concatenate((block1flip, block1ref))

        # Flip the combination
        newblock0flip = np.concatenate((block1flip, block0ref))
        newblock1flip = np.concatenate((block0flip, block1ref))

        minblock0 = refBlock.start
        maxblock0 = refBlock.end
        minblock1 = flipBlock.start
        maxblock1 = flipBlock.end

    return [newblock0, newblock1], [newblock0flip, newblock1flip], [minblock0, maxblock0, minblock1, maxblock1]

def singleParentPhasing(dnmChildBlock, longestBlockparent, genotypes, mutPos, mutCfg, debug_print=False):
    """
    Phase the child's haplotypes based on a single parent's haplotypes.

    Parameters:
    - dnmChildBlock (haplotypeBlock): Child's haploblock containing the DNM.
    - longestBlockparent (haplotypeBlock): Parent's longest haploblock.
    - genotypes (pd.DataFrame): DataFrame containing genotypes.
    - mutPos (int): Position of the mutation.
    - mutCfg (str): Mutation configuration.

    Returns:
    - int or None: Phase value or None if indeterminate.
    """
    commonMinPos = max(longestBlockparent.start, dnmChildBlock.start)
    commonMaxPos = min(longestBlockparent.end, dnmChildBlock.end)

    if commonMaxPos >= commonMinPos:
        # Extract genotypes
        parent = genotypes[genotypes['POS'].between(commonMinPos, commonMaxPos)][longestBlockparent.individual].str.split('|', expand=True)
        child = genotypes[genotypes['POS'].between(commonMinPos, commonMaxPos)]['Child'].str.split('|', expand=True)
        parent0, parent1 = parent[0].astype(int), parent[1].astype(int)

        # Extract mutational configuration
        if mutCfg == '1|0':
            cMut, c0 = child[0].astype(int), child[1].astype(int)
        else:
            c0, cMut = child[0].astype(int), child[1].astype(int)

        if debug_print:
            print(f"c0 : {c0.values}, cMut : {cMut.values}")
            print(f"parent0 : {parent0.values}, parent1 : {parent1.values}")

        if (c0 == cMut).all():
            return None
        else:
            if (c0 == parent1).all() and (cMut == parent0).all():
                return None
            elif (c0 == parent0).all() and (cMut == parent1).all():
                return None
            if (c0 == parent1).all() == (c0 == parent0).all():
                return None
            if (c0 == parent1).all() or (c0 == parent0).all():
                return 0 if longestBlockparent.individual == 'Father' else 1
            elif (cMut == parent1).all() or (cMut == parent0).all():
                return 1 if longestBlockparent.individual == 'Father' else 0
            else:
                return None

def findIndLongestBlock(haploBlocks, mutBlockIdDnm, mutPos):
    """
    Find the individual with the longest haploblock containing the DNM position.

    Parameters:
    - haploBlocks (pd.DataFrame): DataFrame containing haploblocks.
    - mutBlockIdDnm (str): Haploblock ID of the DNM in the child.
    - mutPos (int): Position of the mutation.

    Returns:
    - tuple: Individual identifier and haploblock ID.
    """
    # Get haploblocks for the trio at the mutation position
    HaploBlkChild = haploBlocks[haploBlocks['POS'] == mutPos]['Child'].values[0]
    HaploBlkFather = haploBlocks[haploBlocks['POS'] == mutPos]['Father'].values[0]
    HaploBlkMother = haploBlocks[haploBlocks['POS'] == mutPos]['Mother'].values[0]

    # Calculate sizes of haploblocks
    sizeBlockChild = haploBlocks[haploBlocks['Child'] == HaploBlkChild]['POS'].max() - haploBlocks[haploBlocks['Child'] == HaploBlkChild]['POS'].min()
    sizeBlockFather = haploBlocks[haploBlocks['Father'] == HaploBlkFather]['POS'].max() - haploBlocks[haploBlocks['Father'] == HaploBlkFather]['POS'].min()
    sizeBlockMother = haploBlocks[haploBlocks['Mother'] == HaploBlkMother]['POS'].max() - haploBlocks[haploBlocks['Mother'] == HaploBlkMother]['POS'].min()

    # Find individual with the largest haploblock
    maxsizeBlock = max(sizeBlockChild, sizeBlockFather, sizeBlockMother)

    if maxsizeBlock == sizeBlockChild:
        return 'Child', HaploBlkChild
    elif maxsizeBlock == sizeBlockFather:
        return 'Father', HaploBlkFather
    else:
        return 'Mother', HaploBlkMother

def findCvHaploblocks(haploBlocks, HaploBlkChildlst, HaploBlkFatherlst, HaploBlkMotherlst, mutBlockIdDnm, mutPos):
    """
    Find and create haploblock objects for each individual.

    Parameters:
    - haploBlocks (pd.DataFrame): DataFrame containing haploblocks.
    - HaploBlkChildlst (list): List of child haploblock IDs.
    - HaploBlkFatherlst (list): List of father haploblock IDs.
    - HaploBlkMotherlst (list): List of mother haploblock IDs.
    - mutBlockIdDnm (str): DNM haploblock ID in the child.
    - mutPos (int): Position of the mutation.

    Returns:
    - tuple: Lists of haploblock objects and DNM haploblocks for each individual.
    """
    if not len(haploBlocks):
        return

    # Initialize lists
    allChildBlocks = []
    allFatherBlocks = []
    allMotherBlocks = []

    # Initialize DNM haploblocks
    dnmChildBlock = None
    dnmFatherBlock = None
    dnmMotherBlock = None

    # Dictionary for iteration
    lstIndividuals = {
        'Child': [HaploBlkChildlst, allChildBlocks, dnmChildBlock],
        'Father': [HaploBlkFatherlst, allFatherBlocks, dnmFatherBlock],
        'Mother': [HaploBlkMotherlst, allMotherBlocks, dnmMotherBlock]
    }

    for indiv in lstIndividuals:
        for block in lstIndividuals[indiv][0]:
            block_start = haploBlocks[haploBlocks[indiv] == block]['POS'].min()
            block_end = haploBlocks[haploBlocks[indiv] == block]['POS'].max()
            block_size = block_end - block_start
            dnmpos = block_start <= mutPos <= block_end
            block_obj = haplotypeBlock(
                block_id=block,
                block_size=block_size,
                block_start_pos=block_start,
                block_end_pos=block_end,
                individual=indiv,
                dnmpos=dnmpos
            )
            lstIndividuals[indiv][1].append(block_obj)

            if (block == mutBlockIdDnm) or dnmpos:
                lstIndividuals[indiv][2] = block_obj

    return (lstIndividuals['Child'][1], lstIndividuals['Child'][2],
            lstIndividuals['Father'][1], lstIndividuals['Father'][2],
            lstIndividuals['Mother'][1], lstIndividuals['Mother'][2])

class haplotypeBlock:
    """
    Class to store haploblock information.
    """
    def __init__(self, block_id, block_size, block_start_pos, block_end_pos, individual, dnmpos=False):
        self.id = block_id              # Haploblock ID
        self.size = block_size          # Size of the haploblock
        self.start = block_start_pos    # Start position
        self.end = block_end_pos        # End position
        self.dnm = dnmpos               # Indicates if DNM is contained in the block

        if individual not in ['Father', 'Mother', 'Child']:
            raise ValueError("Individual must be 'Father', 'Mother', or 'Child'")
        self.individual = individual

def phaseDnmBlock(dnmChildBlock, dnmFatherBlock, dnmMotherBlock, haploBlocks, genotypes, mutPos, mutCfg):
    """
    Phase the DNM block using parents' haploblocks.

    Parameters:
    - dnmChildBlock (haplotypeBlock): DNM haploblock in the child.
    - dnmFatherBlock (haplotypeBlock): DNM haploblock in the father.
    - dnmMotherBlock (haplotypeBlock): DNM haploblock in the mother.
    - haploBlocks (pd.DataFrame): DataFrame containing haploblocks.
    - genotypes (pd.DataFrame): DataFrame containing genotypes.
    - mutPos (int): Position of the mutation.
    - mutCfg (str): Mutation configuration.

    Returns:
    - int or None: Phase value or None if indeterminate.
    """
    if not len(genotypes):
        return

    # Exclude the mutation position
    genotypes = genotypes[genotypes['POS'] != mutPos]

    # Find common positions among the trio
    commonMinPos = max(dnmChildBlock.start, dnmFatherBlock.start, dnmMotherBlock.start)
    commonMaxPos = min(dnmChildBlock.end, dnmFatherBlock.end, dnmMotherBlock.end)

    if commonMaxPos >= commonMinPos:
        # Extract genotypes
        mother = genotypes[genotypes['POS'].between(commonMinPos, commonMaxPos)]['Mother'].str.split('|', expand=True)
        father = genotypes[genotypes['POS'].between(commonMinPos, commonMaxPos)]['Father'].str.split('|', expand=True)
        child = genotypes[genotypes['POS'].between(commonMinPos, commonMaxPos)]['Child'].str.split('|', expand=True)
        m0, m1 = mother[0].astype(int), mother[1].astype(int)
        p0, p1 = father[0].astype(int), father[1].astype(int)

        # Extract mutational configuration
        if mutCfg == '1|0':  # c0 carries the mutation
            cMut, c0 = child[0].astype(int), child[1].astype(int)
        else:  # c1 carries the mutation
            c0, cMut = child[0].astype(int), child[1].astype(int)

        # Can't phase if both parents gave the same haplotypes
        if (((c0 == m1).all() or (c0 == m0).all()) and ((c0 == p1).all() or (c0 == p0).all())):
            return None
        if (c0 == cMut).all():  # Can't phase if the child is homozygous
            return None
        else:
            if (c0 == m1).all() or (c0 == m0).all():  # c0 from mother
                return 1  # Mutation from father
            elif (c0 == p1).all() or (c0 == p0).all():  # c0 from father
                return 0  # Mutation from mother
            else:
                return None

# =================================================================================================
# TEST
# =================================================================================================

if __name__ == "__main__":
    # Import individual read-based phases from Whatshap
    ## First case: simple phasing of the DNM haploblock - Mother is the source
    # with open("examples/phasedTrio_1.vcf", 'w') as f: 
    #     f.write("#CHROM\tPOS\t100949\t100934\t100947\n")
    #     f.write("JAKFHU010000086.1\t110\t0|0:3\t0|0:5\t0|1:1\n")
    #     f.write("JAKFHU010000086.1\t111\t1|1:3\t0|0:5\t0|1:1\n")
    #     f.write("JAKFHU010000086.1\t113\t0|1:3\t0|0:5\t0|1:1\n")
    #     f.write("JAKFHU010000086.1\t115\t0|1:4\t1|0:6\t0|0:1\n")
    #     f.write("JAKFHU010000086.1\t116\t0|0:4\t0|0:6\t0|0:.\n")
    
    # ## Second case: phasing of the DNM haploblock with a second longest haploblock - Father is the source
    # with open("examples/phasedTrio_1.vcf", 'w') as f:
    #     f.write("#CHROM\tPOS\t100949\t100934\t100947\n")
    #     f.write("JAKFHU010000086.1\t110\t0|0:3\t0|0:5\t0|1:1\n")
    #     f.write("JAKFHU010000086.1\t111\t0|0:3\t0|0:5\t0|0:1\n")
    #     f.write("JAKFHU010000086.1\t113\t0|1:4\t0|0:6\t1|0:1\n")
    #     f.write("JAKFHU010000086.1\t115\t0|1:4\t0|1:6\t1|1:1\n")
    #     f.write("JAKFHU010000086.1\t116\t0|1:4\t0|1:.\t1|1:1\n")
    
#     # ## Final case: Block chaining method - Mother is the source - Expected Mother to be source of the DNM block
# #     ## Case 5.1: Combine the blocks of the longest haploblocks within the range of the DNM block
    # with open("examples/phasedTrio_1.vcf", 'w') as f:
    #     f.write("#CHROM\tPOS\t100949\t100934\t100947\n")
    #     f.write("JAKFHU010000086.1\t110\t0|0:3\t0|0:6\t0|1:1\n")
    #     f.write("JAKFHU010000086.1\t111\t0|1:3\t0|1:6\t0|1:1\n")
    #     f.write("JAKFHU010000086.1\t113\t0|0:4\t0|1:7\t1|0:1\n")
    #     f.write("JAKFHU010000086.1\t115\t0|0:5\t0|1:7\t1|0:1\n")
    ## Case 5.2 : Combine the blocks of with the longest haploblock in the parents - Expected father to be the source of the DNM 
    # with open("examples/phasedTrio_1.vcf", 'w') as f:
    #     f.write("#CHROM\tPOS\t100949\t100934\t100947\n")
    #     f.write("JAKFHU010000086.1\t110\t0|0:6\t0|0:3\t0|1:1\n")
    #     f.write("JAKFHU010000086.1\t111\t0|1:6\t0|1:3\t0|1:1\n")
    #     f.write("JAKFHU010000086.1\t113\t0|1:7\t0|0:4\t1|0:1\n")
    #     f.write("JAKFHU010000086.1\t115\t0|1:7\t0|0:5\t1|0:1\n")


#     ##  Case 5.3: Child has the longest blockm expected DNM from mum
    with open("examples/phasedTrio_1.vcf", 'w') as f:
        f.write("#CHROM\tPOS\t100949\t100934\t100947\n")
        f.write("JAKFHU010000086.1\t110\t0|0:3\t0|0:6\t0|1:1\n")
        f.write("JAKFHU010000086.1\t111\t0|1:3\t0|1:6\t0|1:1\n")
        f.write("JAKFHU010000086.1\t113\t0|1:4\t1|0:6\t1|0:2\n")
        f.write("JAKFHU010000086.1\t115\t0|1:5\t1|1:6\t1|0:2\n")

    # Genotypes of the trio must be in order: Mother, Father, Child
    phasedVCF = pd.read_csv(
        "examples/phasedTrio_1.vcf",
        comment='#',
        sep="\t",
        names=['CHROM', 'POS', 'Mother', 'Father', 'Child']
    )

    # Mutation positions in bed format
    mutations = pd.read_csv("examples/mutations_1.bed", sep="\t")

    # Form an rbMutationBlock
    print(mutations.loc[0, "CHROM"], mutations.loc[0, "POS"])

    mublock = rbMutationBlock(
        10000,
        mutations.loc[0, 'CHROM'],
        mutations.loc[0, 'POS'],
        phasedVCF,
        debug_print=False
    )

    print(mublock.mutLocus)
    # print(mublock.mutCfg)
    print(mublock.phase)
    print(mublock.phaseMethod)
