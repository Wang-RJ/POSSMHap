import pandas as pd
import numpy as np

# Filter out the rows where POOHA failed

original_data = pd.read_csv('SNP_samplingSNP_all_detail.txt', sep='\t', header=None)
original_data.columns = ["chromosome", "muttype", "position"]
pooha_data = pd.read_csv('POOHA_failed.txt', sep='\t')
# print(original_data.shape)
# print(pooha_data.shape)
df = pd.merge(original_data, pooha_data, on=["chromosome", "position"], how='inner')
df.to_csv("SNP_samplingSNP_all_detail_pooha.txt", header=False, index=False, sep="\t")