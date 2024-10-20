import pandas as pd

def splitGT(df, mut_pos):
    """
    Splits genotype data (GT) from a DataFrame containing VCF format columns ('Mother', 'Father', 'Child').
    Returns two DataFrames: one for genotypes and one for phases.

    Parameters:
    - df (pd.DataFrame): DataFrame containing genotype data with columns 'POS', 'Mother', 'Father', 'Child'.
    - mut_pos (int): Position of the mutation.

    Returns:
    - merged_genos (pd.DataFrame): DataFrame containing genotypes with 'POS' column.
    - merged_phases (pd.DataFrame): DataFrame containing phases with 'POS' column.
    """
    if df.empty:
        return None, None

    # Ensure 'POS' column is numeric
    df['POS'] = pd.to_numeric(df['POS'], errors='coerce')

    # Select relevant columns and create a copy to avoid SettingWithCopyWarning
    vcf_gt = df[['POS', 'Mother', 'Father', 'Child']].copy()

    # Split genotypes and phases
    genos, phases = split_columns(vcf_gt, mut_pos)

    # Merge 'POS' back into genos and phases
    merged_genos = pd.concat([vcf_gt[['POS']].reset_index(drop=True), genos.reset_index(drop=True)], axis=1)
    merged_phases = pd.concat([vcf_gt[['POS']].reset_index(drop=True), phases.reset_index(drop=True)], axis=1)

    return merged_genos, merged_phases


def split_columns(vcf_gt, mut_pos):
    """
    Splits genotype strings by ':' and separates genotypes and phases.
    If the child has haploblock information at the mutation position but the parents do not,
    assigns the child's haploblock to the parents at that position.

    Parameters:
    - vcf_gt (pd.DataFrame): DataFrame with columns 'POS', 'Mother', 'Father', 'Child'.
    - mut_pos (int): Position of the mutation.

    Returns:
    - genos (pd.DataFrame): DataFrame containing genotypes without 'POS' column.
    - phases (pd.DataFrame): DataFrame containing phases without 'POS' column.
    """
    
    
    # List of individuals 
    individuals = ['Mother', 'Father', 'Child']
    # Make a copy of the original df
    genotypes_df = vcf_gt.copy()
    
    # Initialize DataFrames for genotypes and haploblocks
    genos = pd.DataFrame()
    phases = pd.DataFrame()

    # Split genotype strings and separate genotypes and phases
    for individual in individuals:
        # Split by ':'
        split_data = genotypes_df[individual].str.split(':', expand=True)
        
        # First part is genotype
        genos[individual] = split_data[0]
        
        # Second part is phase block ID (if it exists)
        if split_data.shape[1] > 1:
            phases[individual] = split_data[1]
        else:
            phases[individual] = None
            
    # Save the old indexing
    genos_index = genos.index.copy()
    phases_index = phases.index.copy()

    # Set the index to 'POS' column
    genotypes_df = genotypes_df.set_index('POS', drop=True)
    genos = genos.set_index(genotypes_df.index)
    phases = phases.set_index(genotypes_df.index)
    
    # Go through each row 
    for row in genos.index:
        # Extract the haploblock and the genotype
        row_genos = genos.loc[row]
        row_phases = phases.loc[row]    
        
        # Check if haploblock is missing and genotype is homozygous
        for individual in individuals:
            if (row_phases[individual] is None) or (row_phases[individual] == '.'):

                # Find the limits of the blocks contained within the individual phases dataframe
                block_limits = get_min_max_block_index(phases, individual)
                
                # Go through each block
                for b in block_limits:
                    # if we find the block that contains the current position
                    if block_limits[b][0] <= row and row <= block_limits[b][1]:
                        
                        # Assign the haplopblock to the row
                        phases.loc[row, individual] = b
                        row_phases[individual] = b                   
                        
                        # If the haploblock at this row NOT missing
                        if row_phases[individual] is not None:
                            
                            # If the genotype is homozygous, replace '/' with '|'
                            if row_genos[individual] in ['0/0', '1/1']:
                                genos.loc[row, individual] = row_genos[individual].replace('/', '|')
                                
                            # If the genotype is heterozygous and the other parent has a haploblock, replace '/' with '|'            
                            if (row_genos["Father"] in ['0/1', '1/0'])and (not (row_genos["Mother"] in ['0/1', '1/0'])):
                                genos.loc[row, "Father"] = row_genos["Father"].replace('/', '|')
                                genos.loc[row, "Mother"] = row_genos["Mother"].replace('/', '|')
                            elif (row_genos["Father"] not in ['0/1', '1/0'])and ((row_genos["Mother"] in ['0/1', '1/0'])):
                                genos.loc[row, "Father"] = row_genos["Father"].replace('/', '|')
                                genos.loc[row, "Mother"] = row_genos["Mother"].replace('/', '|')
    # Replace '.' in phases with None
    phases.replace('.', None, inplace=True)

    # Assign the mutation position
    mut_row = phases.loc[mut_pos]
    
    # Final fix, if at the haploblock containing the dnm, the parents have no haploblock, assign the child's haploblock to the parents
    if mut_row['Father'] == '.' or mut_row['Mother'] == '.':
       
        # Extract child's genotype and haploblock info at the mutation position
        child_haploblock = mut_row['Child']
        
        # Fill in missing haploblock info for parents if child has it
        if child_haploblock or child_haploblock != '.':
            # Child has haploblock info at the mutation position
            for parent in ['Mother', 'Father']:
                parent_haploblock = mut_row[parent]
                
                if parent_haploblock == '.':
                    # Assign child's haploblock to parent at the mutation position
                    phases.loc[mut_pos, parent] = child_haploblock

    # Return the original indexing
    genos.index = genos_index
    phases.index = phases_index

    return genos, phases

def get_min_max_block_index(phases_df, individual):
    """
    Returns the minimum and maximum indices of the block associated with a chosen individual.

    Parameters:
    - phases_df (pd.DataFrame): DataFrame with columns 'Mother', 'Father', 'Child'.
    - individual (str): The individual to check ('Mother', 'Father', or 'Child').

    Returns:
    - min_index (int): The minimum index where the individual has a block.
    - max_index (int): The maximum index where the individual has a block.
                       Returns None for both if the individual has no blocks.
    """
    # Validate the individual
    if individual not in ['Mother', 'Father', 'Child']:
        raise ValueError("Invalid individual name. Choose 'Mother', 'Father', or 'Child'.")

    # Extract the list of haploblocks
    haploblocks = phases_df[individual].unique()
    result = {}
    for block in haploblocks:
        if block is not None and block != '.':
            block_indices = phases_df[phases_df[individual] == block].index
            result[block] = block_indices.min(), block_indices.max()
    return result

# def spread_haploblocks(phases_df, genos_df, min_max_block_index, individual):
#     individuals = ['Mother', 'Father', 'Child'].pop(individual)
    
#     for block in min_max_block_index:
#         for i in individuals:
#             row_min_phase = phases_df.loc[min_max_block_index[block][0]]
#             row_max_phase = phases_df.loc[min_max_block_index[block][1]]
#             row_min_genos = genos_df.loc[min_max_block_index[block][0]]
#             row_max_genos = genos_df.loc[min_max_block_index[block][1]]
#             if row_min_phase[i] is None:
#                 if row_min_phase[i] in ['0/0', '1/1']:
#                     genos_df.loc[min_max_block_index[block][0], i] = row_min_genos[i].replace('/', '|')
#                     phases_df.loc[min_max_block_index[block][0], i] = block
#             if row_max_phase[i] is None:
#                 if row_max_phase[i] in ['0/0', '1/1']:
#                     genos_df.loc[min_max_block_index[block][1], i] = row_max_genos[i].replace('/', '|')
#                     phases_df.loc[min_max_block_index[block][1], i] = block
                
#     return phases_df, genos_df
        
## =================================================================================================
# # Sample data
# data = {
#     'Chromosome': ['JAKFHU010000160.1'] * 5,
#     'POS': [14494647, 14494743, 14494760, 14495245, 14495569],
#     'Mother': ['0/0:.', '0/0', '0/0:.', '0/0:.', '0/0:.'],
#     'Father': ['0|1:14494647', '0/0', '0|1:14494647', '0|1:14494647', '0|1:14494647'],
#     'Child': ['0/0:.', '0/0:14494648', '0/0:.', '0/0:.', '0/0:.']
# }

# vcf_gt = pd.DataFrame(data)
# print("Input DataFrame:")
# print(vcf_gt)
# # Mutation position
# mut_pos = 14494743


# print("Split GT:")
# genos, phases = splitGT(vcf_gt, mut_pos)
# print("Genotypes:")
# print(genos)
# print("\nPhases:")
# print(phases)