import logging
import pandas as pd
import itertools
import numpy as np
from getMutation import getMutation, make_informative_genotypes, make_informative_haplo_blocks
from splitGT import splitGT
from concatenate import generate_super_matrices

## Update 10/26/2024 - ver 4.

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RbMutationBlock:
    """
    Class to handle phasing of de novo mutations in trios.
    """

    def __init__(self, size, mut_chrom, mut_pos, phase_df=None):
        """
        Initialize an RbMutationBlock object.

        Parameters:
        - size (int): Size of the mutation block.
        - mut_chrom (str): Chromosome where the mutation occurs.
        - mut_pos (int): Position of the mutation on the chromosome.
        - phase_df (pd.DataFrame, optional): DataFrame containing phased genotypes.
        """
        self.size = size
        self.mut_config = None
        self.mut_locus = (mut_chrom, mut_pos)
        self.phase = None         # Phase of the mutation
        self.phase_method = ''    # Method used to phase the mutation
        self.genotypes = None     # Genotypes restricted to the mutation block of the child at the mutation locus
        self.haplo_blocks = None  # Haploblocks for the child, mother, and father
        self.genotypes_unrestricted = None # Genotypes for the trio around the mutation locus, based on the longest haploblock at the mutation locus
        self.haplo_blocks_unrestricted = None # Haploblocks for the trio around the mutation locus, based on the longest haploblock at the mutation locus

        if phase_df is not None:
            self.import_df(phase_df)


    def import_df(self, phase_df):
        """
        Import phase data and process it for phasing.

        Parameters:
        - phase_df (pd.DataFrame): DataFrame containing phased genotypes.
        """
        # Validate input DataFrame
        required_columns = {'CHROM', 'POS', 'Mother', 'Father', 'Child'}
        if not required_columns.issubset(phase_df.columns):
            raise ValueError(f"phase_df must contain the columns: {', '.join(required_columns)}")

        # Step 1: Filter the region around the mutation
        region = self._filter_region(phase_df)

        # Step 2: Extract mutation configuration
        self.mut_config, mut_block_id = self._extract_mutation_config(region)
        if mut_block_id is None:
            logger.debug("Mutation block ID for the Child is None. Exiting import.")
            self.phase = "Missing DNM block"
            return
        
        # Step 3: Identify haploblocks
        genotypes, haplo_blocks = self._identify_haploblocks(region, only_mut=True, longest_individual="Child")
        self.genotypes = genotypes
        self.haplo_blocks = haplo_blocks
        
        logger.debug(f"Mutation at {self.mut_locus}")
        logger.debug(f"Genotypes \n{genotypes}")
        logger.debug(f"Haploblocks \n{haplo_blocks}")
        logger.debug(f"Input data {region}")
        
        # Step 4 : Find individual with the longest haploblock
        genotypes_unrestricted, haplo_blocks_unrestricted = self._identify_haploblocks(region, only_mut=False, longest_individual=None)
        self.genotypes_unrestricted = genotypes_unrestricted
        self.haplo_blocks_unrestricted = haplo_blocks_unrestricted
        
        # Write to csv the genotypes unrestricted and haploblock unrestricted
        # First concatenate genotype and haploblock
        genotypes_unrestricted = genotypes_unrestricted.rename(columns={"Child": "Child_genotype", "Father": "Father_genotype", "Mother": "Mother_genotype"})
        haplo_blocks_unrestricted = haplo_blocks_unrestricted.rename(columns={"Child": "Child_haploblock", "Father": "Father_haploblock", "Mother": "Mother_haploblock"})
        merged_df = pd.merge(genotypes_unrestricted, haplo_blocks_unrestricted, on="POS", suffixes=("_genotype", "_haploblock"))
        merged_df.to_csv(f"gt-hb-{self.mut_locus}.csv", sep="\t")

       
        if genotypes is None or haplo_blocks['Child'].isna().all():
            logger.debug(f"Missing haploblocks for {self.mut_locus} for the Child. Exiting import.")
            return
    
        # Step 5: Perform phasing
        self._perform_phasing(mut_block_id, region)

    
    def _filter_region(self, phase_df):
        """
        Filter the DataFrame to include only the region around the mutation.

        Parameters:
        - phase_df (pd.DataFrame): DataFrame containing phased genotypes.

        Returns:
        - pd.DataFrame: Filtered DataFrame.
        """
        # Mutation location and region size
        mut_chrom, mut_pos = self.mut_locus
        half_size = self.size // 2
        
        # Filter the region of interest
        distance_match = phase_df['CHROM'] == mut_chrom
        position_match = phase_df['POS'].between(mut_pos - half_size, mut_pos + half_size)
        region = phase_df[distance_match & position_match]

        # Make sure mutation position row has valid mutation block if missing
        mutpos_match = region["POS"] == mut_pos
        if region[mutpos_match]["Child"].str.split(":", expand=True)[1].values[0] == ".":
            region.loc[mutpos_match, "Child"] = region.loc[mutpos_match, "Child"].apply(
                lambda x: x.split(":")[0] + ":MUT_BLOCK"
            )
            logger.info(f"Adding MUT_BLOCK to Child. Missing DNM block associated with the De novo positions for: {mut_chrom} {mut_pos}")

        # Get the index of de novo
        mutpos_index = region.index[region["POS"] == mut_pos].tolist()
        mutblock = region.loc[mutpos_index[0], "Child"].split(":")[1]
        
        if mutblock not in [".", None]: 
            # Check for adjacent rows
            prev_index, next_index = mutpos_index[0] - 1,  mutpos_index[0] + 1

            # Check if previous/ next row has homozygous genotype and no block
            if prev_index in region.index:
                child_genotype, child_block = region.loc[prev_index, "Child"].split(":")
                if child_genotype in ["0/0", "0|0"] and child_block in [".", None]:
                        region.at[prev_index, "Child"] = f"{child_genotype}:{mutblock}"
            
            if next_index in region.index:
                child_genotype, child_block = region.loc[next_index, "Child"].split(":")
                if child_genotype in ["1|1", "1/1"]and child_block in [".", None]:
                        region.at[next_index, "Child"] = f"{child_genotype}:{mutblock}"
                    
        return region

    def _extract_mutation_config(self, region):
        """
        Extract the mutation configuration and mutation block ID.

        Parameters:
        - region (pd.DataFrame): DataFrame containing the region around the mutation.

        Returns:
        - tuple: Mutation configuration and mutation block ID.
        """
        # Define mutation location
        _, mut_pos = self.mut_locus
        mut_idx = region.index[region['POS'] == mut_pos]
        
        # Determine mutation configuration based on child's genotype
        if '0/1' in region.loc[mut_idx, "Child"].values[0]:
            mut_config = 0
        elif '1/0' in region.loc[mut_idx, "Child"].values[0]:
            mut_config = 1
            
        # Try to retrieve alternative mutation configuration and block ID   
        mut_config2, mut_block_id = getMutation(region, mut_idx)
        
        # Return alternative configuration if available, otherwise initial configuration
        if mut_config2 is not None:
            return mut_config2, mut_block_id            
        return mut_config, mut_block_id

    def _identify_haploblocks(self, region, only_mut=True, longest_individual=None):
        """
        Identify haploblocks for the trio.

        Parameters:
        - region (pd.DataFrame): DataFrame containing the region around the mutation.
        - only_mut (bool): Flag to indicate if only the mutation block associated with the child is considered
        - haplo_blocks_info (dict, optional): Haploblock information for each individual.
        - longest_individual (str, optional): Individual with the longest haploblock.
        
        Returns:
        - tuple: Processed genotypes and haploblocks DataFrame.
        """
        # Initialize mutation position and split genotypes
        mut_pos = self.mut_locus[1]
        genotypes, haplo_blocks = splitGT(region, mut_pos)
        
        # Determine contained phase positions
        if only_mut:
            contained_positions = self._find_contained_phase(haplo_blocks)
        else:
            contained_positions = self._find_contained_phase(haplo_blocks, longest_individual=longest_individual)
            
        # Filter genotypes and haploblocks
        genotypes = genotypes[genotypes["POS"].isin(contained_positions)]
        haplo_blocks = haplo_blocks[haplo_blocks["POS"].isin(contained_positions)]
        
        # Make informative genotypes and haploblocks
        genotypes = make_informative_genotypes(genotypes, self.mut_locus[1])
        
        # Take all the positions in the haploblocks if there exist at least one non NaN value (any=False), otherwise take only the informative haploblocks
        any = False if longest_individual is None else True
        haplo_blocks = make_informative_haplo_blocks(haplo_blocks, genotypes, contained_positions, any=any) # Not to make the haploblocks informative        
        
        # # Merge genotypes and haplo_blocks for debugging and final processing
        # merged_df = pd.merge(genotypes, haplo_blocks, on="POS", suffixes=("_genotype", "_haploblock"))
        
        # # Save final merged DataFrame with tab delimiter
        # merged_df = pd.merge(genotypes, haplo_blocks, on="POS", suffixes=("_genotype", "_haploblock"))

        return genotypes, haplo_blocks

    def _find_contained_phase(self, haplo_blocks, longest_individual="Child"):
        """
        Find positions contained within the mutation haploblock.

        Parameters:
        - haplo_blocks (pd.DataFrame): DataFrame containing haploblocks.

        Returns:
        - list: List of positions within the mutation haploblock.
        """
        
        if longest_individual is not None:
            mut_block_id = haplo_blocks.loc[haplo_blocks['POS'] == self.mut_locus[1], longest_individual].values[0]
            block_match = haplo_blocks[longest_individual] == mut_block_id
            match_positions = haplo_blocks[block_match]["POS"].tolist()
            min_pos, max_pos = min(match_positions), max(match_positions)
        else:
            haplo_blocks['POS'] = haplo_blocks['POS'].astype(int)
            min_pos, max_pos = haplo_blocks['POS'].min(), haplo_blocks['POS'].max()

        if max_pos > min_pos:
            not_none = haplo_blocks.dropna(subset=['Child', 'Father', 'Mother'], how='all')["POS"]
            contained = set(not_none) & set(range(min_pos, max_pos + 1))
        else:
            contained = {min_pos}
        
        return list(contained)

    def _perform_phasing(self, mut_block_id, region):
        """
        Perform phasing using different methods.

        Parameters:
        - mut_block_id (str): Mutation block ID in the child.
        """
        
        mut_pos = self.mut_locus[1]
        logger.info(f"\n=========Performing phasing for mutation at {self.mut_locus}=========\n")
                
        # Method 1: Phase using DNM haploblock method
        logger.info("Method 1: Phasing using DNM haploblock method")
        longest_individual, longest_block_id = self._find_longest_block(mut_block_id)
        haplo_blocks_info = self._find_haplo_blocks(mut_block_id)
        phase = self._phase_dnm_block(haplo_blocks_info)
        
        if phase is not None:
            self.phase = self._get_explicit_phase(phase)
            self.phase_method = 'DNM Haploblock'
            logger.info(f"Found phase: {self.phase} using method: {self.phase_method}")
            return
        else:
            logger.info(f"Phasing using DNM haploblock unsuccessful.")

        # Method 2: Attempt phasing using longest haploblock in parents
        logger.info("Method 2: Attempt phasing using longest haploblock in parents")        
        phase, parentID, parent_haploblock = self._phase_using_longest_haploblock(longest_individual, haplo_blocks_info)
        
        if phase is not None:
            self.phase = self._get_explicit_phase(phase)
            self.phase_method = 'Longest Haploblock Phasing'
            logger.info(f"Found phase: {self.phase} using method: {self.phase_method} for mutation at {self.mut_locus}")
            logger.info(f"Parent used for phasing: {parentID} and Haploblock: {parent_haploblock}")
            return
        else:
            logger.info(f"Phasing using longest haploblock in parents unsuccessful.")

        # Method 3: Attempt block chaining method
        logger.info("Method 3: Attempt block chaining method")
     
        # Sort the blocks by their starting position
        haplo_blocks_child = [b for b in haplo_blocks_info['Child']['blocks']] + [haplo_blocks_info['Child']['dnm_block']] #.sort(key=lambda x: x.start)
        haplo_blocks_father = [b for b in haplo_blocks_info['Father']['blocks']] + [haplo_blocks_info['Father']['dnm_block']] 
        haplo_blocks_mother = [b for b in haplo_blocks_info['Mother']['blocks']] + [haplo_blocks_info['Mother']['dnm_block']]
        
        logger.info(f"Number of combinable blocks for father is {len(haplo_blocks_father)}")
        logger.info(f"Number of combinable blocks for mother is {len(haplo_blocks_mother)}")
        logger.info(f"Number of combinable blocks for child is {len(haplo_blocks_child)}")
        
        logger.info(f"The IDs of combinable blocks for father is {[b.id for b in haplo_blocks_father]}")
        logger.info(f"The IDs of combinable blocks for mother is {[b.id for b in haplo_blocks_mother]}")
        logger.info(f"The IDs of combinable blocks for child is {[b.id for b in haplo_blocks_child]}")
        
        if len(haplo_blocks_child) > 5 or len(haplo_blocks_father) > 5 or len(haplo_blocks_mother) > 5:
            k = 1
            while k < min(len(haplo_blocks_child) - 1, len(haplo_blocks_father) - 1, len(haplo_blocks_mother) - 1):
                logger.info(f"Restricting combinations to {k+1} haploblocks/individual. Including the mutation block")
                
                # Restrict to 5 haploblocks
                haplo_blocks_child_restrict = haplo_blocks_child[:k] + [haplo_blocks_child[-1]]
                haplo_blocks_father_restrict = haplo_blocks_father[:k] + [haplo_blocks_father[-1]]
                haplo_blocks_mother_restrict = haplo_blocks_mother[:k] + [haplo_blocks_mother[-1]]
            
                # # Sort the blocks by their starting position
                # haplo_blocks_mother_sorted = sorted(haplo_blocks_mother_restrict, key=lambda x: x.start)
                # haplo_blocks_father_sorted = sorted(haplo_blocks_father_restrict, key=lambda x: x.start)
                # haplo_blocks_child_sorted = sorted(haplo_blocks_child_restrict, key=lambda x: x.start)
                
                # Sort the blocks by their relative distance to the mutation ind
                haplo_blocks_child_sorted = sorted(haplo_blocks_child_restrict, key=lambda x: abs(x.start - mut_pos))
                haplo_blocks_father_sorted = sorted(haplo_blocks_father_restrict, key=lambda x: abs(x.start - mut_pos))
                haplo_blocks_mother_sorted = sorted(haplo_blocks_mother_restrict, key=lambda x: abs(x.start - mut_pos))                
                
                # Print out the blocks used
                logger.info(f"Blocks IDs used for child: {[b.id for b in haplo_blocks_child_restrict]}")
                logger.info(f"Blocks IDs used for father: {[b.id for b in haplo_blocks_father_restrict]}")
                logger.info(f"Blocks IDs used for mother: {[b.id for b in haplo_blocks_mother_restrict]}")
                
                logger.info(f"Number of blocks considered - Child: {len(haplo_blocks_child_restrict)}, Father: {len(haplo_blocks_father_restrict)}, Mother: {len(haplo_blocks_mother_restrict)}")
            
                # Attempt phasing with restricted blocsk
                phase, stop_combining  = self._attempt_block_chaining(
                    haplo_blocks_child=haplo_blocks_child_sorted, 
                    haplo_blocks_father=haplo_blocks_father_sorted,
                    haplo_blocks_mother=haplo_blocks_mother_sorted
                )
                
                if phase is not None and stop_combining == False:
                    self.phase = self._get_explicit_phase(phase)
                    self.phase_method = 'Block Chaining Method'
                    logger.info(f"Found phase: {self.phase} using method: {self.phase_method} with {k} blocks")
                    return
                
                
                # If number of blocks reached
                if stop_combining == True:
                    logger.info(f"Phasing unsuccessful. Limit number of combined blocks (200) reached. Number of singular blocks used {k}")
                    return
                logger.info(f"Phasing unsuccessful. Found confusing phases. Number of block used {k}")
                if k == 3:
                    break
                k += 1           
        else:
            # Sort all haploblocks if restriction is unnecessary
            haplo_blocks_child_sorted = sorted(haplo_blocks_child, key=lambda x: x.start)
            haplo_blocks_father_sorted = sorted(haplo_blocks_father, key=lambda x: x.start)
            haplo_blocks_mother_sorted = sorted(haplo_blocks_mother, key=lambda x: x.start)
        
            phase = self._attempt_block_chaining(
                haplo_blocks_child=haplo_blocks_child_sorted, 
                haplo_blocks_father=haplo_blocks_father_sorted,
                haplo_blocks_mother=haplo_blocks_mother_sorted
            )
            
            if phase is not None:
                self.phase = self._get_explicit_phase(phase)
                self.phase_method = 'Block Chaining Method'
                logger.info(f"Found phase: {self.phase} using method: {self.phase_method}.")
                return

            logger.info(f"Phasing unsuccessful.")

    def _calculate_phasing_distances(self, child_mut_hap, child_other_hap, mother_alleles, father_alleles):
        """
        Calculate Hamming distances for phasing configurations.

        Parameters:
        - child_mut_hap (pd.Series): Child haplotype carrying the mutation.
        - child_other_hap (pd.Series): Child haplotype not carrying the mutation.
        - mother_alleles (pd.DataFrame): Mother's haplotypes.
        - father_alleles (pd.DataFrame): Father's haplotypes.

        Returns:
        - dict: Distances for different configurations.
        """
        def hamming_distance(x, y):
            return np.sum(x != y)
       
        configurations = {
            'maternal': [
                hamming_distance(child_mut_hap, mother_alleles[i]) + hamming_distance(child_other_hap, father_alleles[j]) \
                    for i in range(2) for j in range(2)
            ],
            'paternal': [hamming_distance(child_other_hap, mother_alleles[i]) + hamming_distance(child_mut_hap, father_alleles[j]) \
                for i in range(2) for j in range(2)
            ]
        }
        return configurations

    def _decide_phase(self, distances, min_support, max_distance, mutation_on_c0):
        """
        Decide the phase based on distances.

        Parameters:
        - distances (dict): Distances for different configurations.
        - min_support (int): Minimum support for phasing.
        - max_distance (int): Maximum allowable Hamming distance.
        - mutation_on_c0 (bool): Indicates if mutation is on c0.

        Returns:
        - int or None: Phase value.
        """
        maternal_distance = distances['maternal']
        paternal_distance = distances['paternal']
    
        # logger.info(f"Mutation on c0 : {mutation_on_c0}")
        if abs(np.min(maternal_distance) - np.min(paternal_distance)) < min_support:
            logger.debug("Ambiguous phasing due to small difference in distances.")
            return None
       
        if min(maternal_distance) < max_distance:
            phase = 0  # match between c0 with maternal parent
        elif min(paternal_distance) < max_distance:
            phase = 1  # match between c0 with paternal parent
        else:
            logger.debug("No reliable phase found due to large distances.")
            return None
        
        # Adjust the phase based on the mutation configuration
        if not mutation_on_c0: # Mutation is on c1
            phase = 1 - phase
        return phase

    def _find_longest_block(self, mut_block_id):
        """
        Find the individual with the longest haploblock containing the mutation.

        Parameters:
        - mut_block_id (str): Mutation block ID in the child.

        Returns:
        - tuple: Individual identifier and haploblock ID.
        """
        haplo_blocks = self.haplo_blocks
        mut_pos = self.mut_locus[1]

        # Get haploblock IDs at the mutation position
        haplo_blk_child = haplo_blocks[haplo_blocks['POS'] == mut_pos]['Child'].values[0]
        haplo_blk_father = haplo_blocks[haplo_blocks['POS'] == mut_pos]['Father'].values[0]
        haplo_blk_mother = haplo_blocks[haplo_blocks['POS'] == mut_pos]['Mother'].values[0]

        # Calculate sizes of haploblocks
        size_block_child = self._get_block_size(haplo_blocks, 'Child', haplo_blk_child)
        size_block_father = self._get_block_size(haplo_blocks, 'Father', haplo_blk_father)
        size_block_mother = self._get_block_size(haplo_blocks, 'Mother', haplo_blk_mother)

        # Find individual with the largest haploblock
        sizes = {
            'Child': size_block_child,
            'Father': size_block_father,
            'Mother': size_block_mother
        }
        longest_individual = max(sizes, key=sizes.get)
        
        longest_block_id = {
            'Child': haplo_blk_child,
            'Father': haplo_blk_father,
            'Mother': haplo_blk_mother
        }[longest_individual]

        return longest_individual, longest_block_id

    def _get_block_size(self, haplo_blocks, individual, block_id):
        """
        Calculate the size of a haploblock.

        Parameters:
        - haplo_blocks (pd.DataFrame): DataFrame containing haploblocks.
        - individual (str): Individual identifier.
        - block_id (str): Haploblock ID.

        Returns:
        - int: Size of the haploblock.
        """
        positions = haplo_blocks[haplo_blocks[individual] == block_id]['POS'].values
        return len(positions)

    def _find_haplo_blocks(self, mut_block_id):
        """
        Find and create haploblock objects for each individual.

        Parameters:
        - mut_block_id (str): Mutation block ID in the child.

        Returns:
        - dict: Haploblock information for each individual.
        """
        haplo_blocks = self.haplo_blocks_unrestricted
        mut_pos = self.mut_locus[1]
        individuals = ['Child', 'Father', 'Mother']
        haplo_info = {}

        for individual in individuals:
            block_ids = haplo_blocks[individual].dropna().unique()
            blocks = []

            for block_id in block_ids:
                block_positions = haplo_blocks[haplo_blocks[individual] == block_id]['POS'].values
                block_start = block_positions.min()
                block_end = block_positions.max()
                block_size = block_end - block_start + 1
                contains_mutation = (block_id == mut_block_id) or (block_start <= mut_pos <= block_end)
                
                block_obj = HaplotypeBlock(
                    block_id=block_id,
                    block_size=block_size,
                    block_start_pos=block_start,
                    block_end_pos=block_end,
                    individual=individual,
                    contains_mutation=contains_mutation,
                    positions=block_positions.tolist(),
                    mut_pos=mut_pos
                )
                blocks.append(block_obj)

            haplo_info[individual] = {
                'blocks': [b for b in blocks if not b.contains_mutation],
                'dnm_block': next((b for b in blocks if b.contains_mutation), None)
            }
        return haplo_info

    def _phase_dnm_block(self, haplo_info):
        """
        Phase using the DNM haploblock method.

        Parameters:
        - haplo_info (dict): Haploblock information for each individual.

        Returns:
        - int or None: Phase value.
        """
        child_block = haplo_info['Child']['dnm_block']
        father_block = haplo_info['Father']['dnm_block']
        mother_block = haplo_info['Mother']['dnm_block']

        if not all([child_block, father_block, mother_block]):
            return None
        
        # Find common positions among the trio
        common_positions = set(child_block.positions).intersection(
            father_block.positions, mother_block.positions
        )
        if not common_positions:
            return None

        # Exclude the mutation position
        genotypes = self.genotypes[self.genotypes['POS'] != self.mut_locus[1]]
        genotypes = genotypes[genotypes['POS'].isin(common_positions)]
        
        if genotypes.empty:
            return None
        
        logger.info(f"The dnm block used {genotypes}")
        logger.info(f"Haploblock used {self.haplo_blocks}")

        # Extract haplotypes and filter informative genotypes
        columns_to_check = ['Mother', 'Father', 'Child']
        filtered_genotypes = genotypes.loc[
            genotypes[columns_to_check].apply(
                lambda col: col.str.split('|', expand=True).apply(lambda x: x.isin(['0', '1']).all(), axis=1)
            ).all(axis=1)
        ]
        
        mother_alleles = filtered_genotypes['Mother'].str.split('|', expand=True).astype(int)
        father_alleles = filtered_genotypes['Father'].str.split('|', expand=True).astype(int)
        child_alleles = filtered_genotypes['Child'].str.split('|', expand=True).astype(int)

        # Determine which haplotype carries the mutation
        mutation_on_c0 = (self.mut_config == 1)
        logger.info(f"Mutation on c0: {mutation_on_c0}")
        child_mut_hap, child_other_hap = (child_alleles[0], child_alleles[1]) # if mutation_on_c0 else (child_alleles[1], child_alleles[0])
        
        # Check if child haplotypes match parents
        if (child_other_hap == child_mut_hap).all():
            return None  # Can't phase if the child is homozygous

        # Determine phase
        distances = self._calculate_phasing_distances(
            child_other_hap=child_other_hap, child_mut_hap=child_mut_hap, 
            mother_alleles=mother_alleles, father_alleles=father_alleles
        )
        
        # logger.info(f"The distances are {distances}")

        return self._decide_phase(distances, min_support=1, max_distance=1, mutation_on_c0=mutation_on_c0)

    def _phase_using_longest_haploblock(self, longest_individual, haplo_info):
        """
        Phase using the longest haploblock in the parents.

        Parameters:
        - longest_individual (str): Individual with the longest haploblock.
        - haplo_info (dict): Haploblock information for each individual.

        Returns:
        - int or None: Phase value.
        """
        dnm_child_block = haplo_info['Child']['dnm_block']
        
        if longest_individual == 'Child':
            # Get blocks for both parents and sort by size
            parent_blocks = haplo_info['Father']['blocks'] + haplo_info['Mother']['blocks']
            parent_blocks.sort(key=lambda x: x.size, reverse=True)
            
            for parent_block in parent_blocks:
                logger.debug(f"Phasing using longest block from {parent_block.individual}: {parent_block.id}")
                phase = self._single_parent_phasing(dnm_child_block, parent_block)
                if phase is not None:
                    return phase, parent_block.individual, parent_block.id
        else:
            # Get child blocks without mutation and sort by size
            child_blocks = haplo_info['Child']['blocks']
            child_blocks = [b for b in child_blocks if not b.contains_mutation]
            child_blocks.sort(key=lambda x: x.size, reverse=True)
            parent_block = haplo_info[longest_individual]['dnm_block']

            for child_block in child_blocks:
                phase = self._single_parent_phasing(child_block, parent_block)
                if phase is not None:
                    return phase, parent_block.individual, parent_block.id
        return None, None, None

    def _single_parent_phasing(self, child_block, parent_block):
        """
        Attempt phasing using a single parent's haploblock.

        Parameters:
        - child_block (HaplotypeBlock): Child's haploblock.
        - parent_block (HaplotypeBlock): Parent's haploblock.

        Returns:
        - int or None: Phase value.
        """
        common_positions = set(child_block.positions).intersection(parent_block.positions)
        if not common_positions:
            return None

        genotypes = self.genotypes[self.genotypes['POS'].isin(common_positions)]
        child_alleles = genotypes['Child'].str.split('|', expand=True).astype(int)
        parent_alleles = genotypes[parent_block.individual].str.split('|', expand=True).astype(int)

        # Determine which haplotype carries the mutation
        mutation_on_c0 = (self.mut_config == 1)
        child_mut_hap, child_other_hap = child_alleles[0], child_alleles[1]

        if (child_other_hap == child_mut_hap).all():
            return None  # Cannot phase if the child is homozygous
        
        # Calculate Hamming distances
        phase = self._calculate_single_parent_phasing_distances(
            child_mut_hap=child_mut_hap, child_other_hap=child_other_hap,
            parent_alleles=parent_alleles, mutation_on_c0=mutation_on_c0, 
            parent_id=parent_block.individual, common_positions = common_positions
        ) 
        return phase
    
    def _calculate_single_parent_phasing_distances(self, child_mut_hap, child_other_hap, parent_alleles, parent_id, mutation_on_c0, common_positions):
        """
        Calculate Hamming distances for phasing with a single parent.

        Parameters:
        - child_mut_hap (pd.Series): Child haplotype carrying the mutation.
        - child_other_hap (pd.Series): Child haplotype not carrying the mutation.
        - parent_alleles (pd.DataFrame): Parent's haplotypes.

        Returns:
        - dict: Distances for different configurations.
        """
        
        logger.debug(f"Current parent id is {parent_id}")
        
        def hamming_distance(x, y):
            return np.sum(x != y)
        
        configurations = {
            'c0_from_parent': [hamming_distance(child_mut_hap, parent_alleles[0]), hamming_distance(child_mut_hap, parent_alleles[1])],
            'c1_from_parent': [hamming_distance(child_other_hap, parent_alleles[0]), hamming_distance(child_other_hap, parent_alleles[1])]
        }
        
        logger.debug(f"Configurations from parent {parent_id} are: ", configurations)
        
        # If no perfect match return None
        if min(configurations['c0_from_parent']) > 0  and sum(configurations['c1_from_parent']) > 0:
            return 
        
        # If there is two perfect matches
        if min(configurations['c0_from_parent']) == 0  and min(configurations['c1_from_parent']) == 0:  # When the parent we're looking at can give both haplotypes to the child's haplotypes
            
            # Check if the other parent is homozygous. If it is the case, then we can infer the haplotype of the child (even without the haploblock associated with the other parent)
            other_parent = "Mother" if parent_id == "Father" else "Father"
            gt_otherparent = self.genotypes[self.genotypes["POS"].isin(common_positions)][other_parent]
            
            # Assign the haplotype to the other parent if they are homozyous
            otherparent_hap = np.zeros(len(common_positions)) if list(gt_otherparent.unique()) in [["0/0"], ["0|0"]] else np.ones(len(common_positions))
                
            if hamming_distance(child_mut_hap, otherparent_hap) == 0:
                config_code = 1 if hamming_distance(child_other_hap, parent_alleles[0]) == 0 else 0
            elif hamming_distance(child_other_hap, otherparent_hap) == 0:
                config_code = 0 if hamming_distance(child_mut_hap, parent_alleles[0]) == 0 else 1
            else:
                logger.debug("Not assignable from this block of other parent {otherparent_hap} at pos: {common_pos}")
                return


        # If we can have an uniquely identifiable happlotype
        elif min(configurations['c0_from_parent']) == 0 and min(configurations['c1_from_parent']) > 0:
            config_code = 0
        elif min(configurations['c1_from_parent']) == 0 and min(configurations['c0_from_parent']) > 0:
            config_code = 1
        else:
            return 

        # Decide if c0 has the mutation
        # if c0 comes from the parent and has the mutation, the phase is the parent_code
        # if c0 comes from the parent and does not have the mutation, the phase is 1 - parent_code
        temp = config_code ^ int(mutation_on_c0)
        return int(temp == (1 if parent_id == 'Father' else 0))    

    def _attempt_block_chaining(self, haplo_blocks_child, haplo_blocks_father, haplo_blocks_mother):
        """
        Attempt phasing using the block chaining method.

        Parameters:
        - longest_individual (str): Individual with the longest haploblock.
        - haplo_info (dict): Haploblock information for each individual.

        Returns:
        - int or None: Phase value.
        
        """
        stop_combine = False
        # Make a copy of the genotypes and haploblocks
        genotypes_unrestricted = self.genotypes_unrestricted.copy()
        genotypes_unrestricted.set_index('POS', inplace=True)
        genotypes_unrestricted.drop(self.mut_locus[1], inplace=True)
        
        # Extract positions
        positions_child = list(itertools.chain(*[b.positions for b in haplo_blocks_child]))
        positions_father = list(itertools.chain(*[b.positions for b in haplo_blocks_father]))
        positions_mother = list(itertools.chain(*[b.positions for b in haplo_blocks_mother]))
        common_positions = set(positions_child).intersection(positions_father, positions_mother)
        logger.info(f"The positions used for assigning mutations: {common_positions}")

        list_blocks_father = [genotypes_unrestricted[genotypes_unrestricted.index.isin(set(haplo_blocks_father[k].positions) & common_positions)]['Father'].str.split('|', expand=True).astype(int).to_numpy() for k in range(len(haplo_blocks_father))]
        list_blocks_mother = [genotypes_unrestricted[genotypes_unrestricted.index.isin(set(haplo_blocks_mother[k].positions) & common_positions)]['Mother'].str.split('|', expand=True).astype(int).to_numpy() for k in range(len(haplo_blocks_mother))]
        list_blocks_child = [genotypes_unrestricted[genotypes_unrestricted.index.isin(set(haplo_blocks_child[k].positions) & common_positions)]['Child'].str.split('|', expand=True).astype(int).to_numpy() for k in range(len(haplo_blocks_child))]
        
        # Determine which haplotype carries the mutation
        mutation_on_c0 = (self.mut_config == 1)
        dnm_block_child = list_blocks_child[[k for k, b in enumerate(haplo_blocks_child) if b.contains_mutation][0]]
        
        ## Create sublist of haploblocks in increasing order of their relative distance to the mutation
    
        combined_blocks_child = generate_super_matrices(list_blocks_child, M_star=dnm_block_child)
        combined_blocks_father = generate_super_matrices(list_blocks_father, M_star=list_blocks_father[0])
        combined_blocks_mother = generate_super_matrices(list_blocks_mother, M_star=list_blocks_mother[0])
        
       
        if len(combined_blocks_child) > 200 or len(combined_blocks_father) > 200 and len(combined_blocks_mother) > 200:
            logger.info("Too many combined block")
            stop_combine = True
            return None, stop_combine

        all_phases = []
        logger.info("Done generating combined blocks for each individuals, preparing to generate combinations....")
        logger.info(f"Number of combined blocks in the child is {len(combined_blocks_child)}")
        logger.info(f"Number of combined blocks for the mother is {len(combined_blocks_mother)}")
        logger.info(f"Number of combined blocks for the father is {len(combined_blocks_father)}")
        print()
        print("Start to assign mutations")
        
        for b_c in combined_blocks_child:
            for b_m in combined_blocks_mother:
                for b_f in combined_blocks_father:
                    print("Child: ", b_c)
                    print("Mother: ", b_m)
                    print("Father: ", b_f)
                    distances = self._calculate_phasing_distances(
                        child_mut_hap=pd.DataFrame(b_c)[0], child_other_hap=pd.DataFrame(b_c)[1],
                        mother_alleles=pd.DataFrame(b_m), father_alleles=pd.DataFrame(b_f)
                    )
                    print("Distances: ", distances)
                    phase = self._decide_phase(distances, 1, 1, mutation_on_c0)
                    all_phases.append(phase if phase is not None else 1000)
                    print(all_phases)
                    if 0 in all_phases and 1 in all_phases:
                        return None, False
        # logger.info(f"All possible phases {all_phases}")
        if 1 not in all_phases and 0 in all_phases:
            return 0, False
        elif 0 not in all_phases and 1 in all_phases:
            return 1, False
        
           

    def _get_explicit_phase(self, phase):
        """
        Convert numerical phase value to explicit 'Maternal' or 'Paternal'.

        Parameters:
        - phase (int): Numerical phase value.

        Returns:
        - str: 'Maternal' or 'Paternal' based on the phase value.
        """
        phase_values = ['Maternal', 'Paternal']
        return phase_values[phase] if phase is not None else None

class HaplotypeBlock:
    """
    Class to store haploblock information.
    """

    def __init__(self, block_id, block_size, block_start_pos, block_end_pos, individual,
                 contains_mutation=False, positions=None, mut_pos=None):
        """
        Initialize a HaplotypeBlock object.

        Parameters:
        - block_id (str): Haploblock ID.
        - block_size (int): Size of the haploblock.
        - block_start_pos (int): Start position of the haploblock.
        - block_end_pos (int): End position of the haploblock.
        - individual (str): Individual identifier ('Father', 'Mother', 'Child').
        - contains_mutation (bool): Indicates if DNM is contained in the block.
        - positions (list): Positions in the haploblock.
        """
        self.id = block_id
        self.size = block_size
        self.start = block_start_pos
        self.end = block_end_pos
        self.individual = individual
        self.contains_mutation = contains_mutation
        self.positions = positions or []

        # Relative positions compared to the dnm position
        if (mut_pos is not None):
            self.relative = int(abs((self.end - self.start)/2 - mut_pos))
            
        else: 
            self.relative = None
        if individual not in ['Father', 'Mother', 'Child']:
            raise ValueError("Individual must be 'Father', 'Mother', or 'Child'")

# ============================================================================================= #
# Example Usage (For Testing)
# ============================================================================================= #
# if __name__ == "__main__":    
    # pass
    # Main data
    # ----------------------------------------------------------------------------------------- #
    # Import individual read-based phases from Whatshap
    # phased_vcf = pd.read_csv(
    #     "examples/test.vcf",
    #     comment='#',
    #     sep="\t",
    #     # names=['CHROM', 'POS', 'Mother', 'Father', 'Child']
    #     names = ['CHROM', 'POS', 'Child', 'Father', 'Mother']
    # )

    # # # Mutation positions in bed format
    # mutations = pd.read_csv("examples/mutations_testvcf.bed", sep="\t")
    
    # # ## ----------------------------------------------------------------------------------------- #
    # # Test on one mutation
    # k = 70
    # ck = RbMutationBlock(10000,
    #                         mutations.loc[k, 'CHROM'],
    #                         mutations.loc[k, 'POS'],
    #                         phased_vcf)

    # # Check position 32, 12, 79, 70, 8


    # ## ----------------------------------------------------------------------------------------- #
    # # # Calculate phase for every mutation
    # phase_blocks = [
    #     RbMutationBlock(10000, chrom, pos, phased_vcf)
    #     for chrom, pos in zip(mutations["CHROM"], mutations["POS"])
    # ]

    # # Extract phases
    # mu_phases = [
    #     (block.mut_locus, block.phase, block.phase_method)
    #     for block in phase_blocks if block.phase in ["Maternal", "Paternal"]
    # ]
    
    # not_phased = [
    #     (block.mut_locus, block.phase, block.phase_method)
    #     for block in phase_blocks if block.phase not in ["Maternal", "Paternal"]
    # ]

    # print("Phased Mutations:")
    # for locus, phase, method in mu_phases:
    #     print(f"Mutation at {locus}: Phase - {phase}, Method - {method}")
    # print(f"Number of mutations phased: {len(mu_phases)}")
    
    # print("Not Phased Mutations:")
    # for locus, phase, method in not_phased:
    #     print(f"Mutation at {locus}: Phase - {phase}, Method - {method}")
        
    # print(f"Number of mutations not phased: {len(not_phased)}")
