import pandas as pd


## ============================================================================================= ##
# Given a dataframe of genotypes and block IDs
# returns the genotype and block ID for a mutation in the individual
def getMutation(df, idx, individual='Child'):
    """
    Get the genotype and block ID for a mutation in the child from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing phase data. 
        idx (int): Index of the mutation in the DataFrame
        
    Returns:
        tuple: A tuple containing the genotype and block ID for the mutation in the child.
    """
    if len(df) < 1:
        return None, None
    
    # a phased genotype entry in a vcf has two elements split by a colon   
    entry = df.loc[idx, individual].item().split(':')

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
