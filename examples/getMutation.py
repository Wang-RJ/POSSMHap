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
        cfg = 1
    elif geno == '0|1':
        cfg = 0
    else:
        cfg = None
        
    return cfg, block    


# Turn each individual's genotypes into an 2-column array of informative sites
def make_informative_genotypes(genotypes):
    # Give heterozygotes in parents a phase if it's the only transmittable genotype,
    # This allows us to phase e.g.,
    #           Mother    Father    Child
    #           0/0       0/1         0|1
    for index, row in genotypes.iterrows():
        if (row[['Mother', 'Father']] == '0/1').sum() == 1:
            genotypes.loc[index] = row.replace('0/1', '0|1', regex=True)
            
    genotypes = genotypes.replace({
        '1/1': '1|1',
        '0/0': '0|0',
        '0/1': None, 
        '0/2': None,
        '2/0': None,
        '1/2': None,
        '2/1': None
        })

    # Count homozygote genotypes as phase informative, drop unphased heterozygotes
    genotypes = genotypes.dropna()
    return genotypes

def make_informative_haplo_blocks(haplo_blocks, genotypes, contained_positions):
    # for each row in the haploblocks, check if cell is None
    # if it is, replace with the corresponding genotype
    
    parents = ['Mother', 'Father']
    for i in set(contained_positions) & set(genotypes['POS']):  
        for j in range(2):

            # if the cell is None, replace with the corresponding genotype
            if haplo_blocks[haplo_blocks['POS'] == i][parents[j]].values[0] is None:
                if genotypes[genotypes['POS'] == i][parents[j]].values[0] in ["0|0", "1|1"]:
                    haplo_blocks.loc[haplo_blocks['POS'] == i, parents[j]] = haplo_blocks[haplo_blocks['POS'] == i]['Child']
        
                if (genotypes[genotypes['POS'] == i][parents] == '0|1').values.sum() == 1:
                    haplo_blocks.loc[haplo_blocks['POS'] == i, 'Mother'] = haplo_blocks.loc[haplo_blocks['POS'] == i, 'Child'].values
                    haplo_blocks.loc[haplo_blocks['POS'] == i, 'Father'] = haplo_blocks[haplo_blocks['POS'] == i]['Child'].values

    haplo_blocks = haplo_blocks.dropna()
    return haplo_blocks          
    
    