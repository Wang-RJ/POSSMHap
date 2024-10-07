import pandas as pd

# ## ============================================================================================= ##
def splitGT(df):
    """
    Splits genotype data (GT) from a DataFrame containing VCF format columns (Mother, Father, Child).
    Returns two DataFrames: one for genotypes and one for phases.
    """
    
    def splitColumns(vcfGT):
        """
        Utility function to split genotype strings by colon and separate genotypes and phases.
        Replaces '.' in phases with None.
        """
        genos, phases = pd.DataFrame(), pd.DataFrame()
        
        for individual in vcfGT:
            # Split columns by ':' and expand into multiple columns
            newCols = vcfGT[individual].str.split(':', expand=True)
            
            # Store genotype (first part) and block ID(second part if it exists)
            
            genos[individual] = newCols[0]
            phases[individual] = newCols[1] if len(newCols.columns) > 1 else None

        # Replace '.' in phases with None and return genos and cleaned phases
        return genos, phases.replace('.', None)
    
    
    # If the input DataFrame is empty, return None for both outputs
    if df.empty:
        return None, None

    # Make sure that the POS column is numeric
    df['POS'] = pd.to_numeric(df['POS'], errors='coerce')
    
    # Split the DataFrame into genotypes and phases dataframes
    genos, phases = splitColumns(df[['Mother', 'Father', 'Child']])
    merged_genos = pd.merge(df[['POS']], genos, left_index=True, right_index=True)
    merged_phases = pd.merge(df[['POS']], phases, left_index=True, right_index=True)
    return merged_genos, merged_phases
