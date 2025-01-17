import pandas as pd
import argparse
from collections import defaultdict
import re

# def parse_vcf(file_name, chromosome, position, child_column_index):
#     # Initialize data storage
#     extracted_data = []
#     previous_row = None
#     target_row = None
#     next_row = None

    
#     # Open the file and process it line by line
#     with open(file_name, 'r') as file:
#         lines = file.readlines()
#         # Remove the header line
#         # lines = lines[1:]
  
#         for i, line in enumerate(lines):
#             # Skip header lines containing #CHROM or other metadata
#             if line.startswith("#"):
#                 continue
#             if line.strip():  # Skip empty lines
#                 cols = line.strip().split('\t')
#                 current_chrom = cols[0]
#                 current_pos = int(cols[1])
                
#                 # Check if the line matches the chromosome and position
#                 if current_chrom == chromosome and current_pos == position:
#                     # Extract data for the current row
#                     ref = cols[2]
#                     alt = cols[3]
#                     child_column_data = cols[child_column_index]
#                     genotype, haploblock = child_column_data.split(':') if ':' in child_column_data else (child_column_data, None)
#                     target_row = {
#                         'Chromosome': current_chrom,
#                         'Position': current_pos,
#                         'Reference': ref,
#                         'Alternative': alt,
#                         'Genotype': genotype,
#                         'Haploblock': haploblock
#                     }
                    
#                     # Check for the previous row
#                     if i > 0:
#                         prev_cols = lines[i - 1].strip().split('\t')
#                         prev_ref = prev_cols[2]
#                         prev_alt = prev_cols[3]
#                         prev_child_data = prev_cols[child_column_index]
#                         prev_genotype, prev_haploblock = prev_child_data.split(':') if ':' in prev_child_data else (prev_child_data, None)
#                         previous_row = {
#                             'Chromosome': prev_cols[0],
#                             'Position': int(prev_cols[1]),
#                             'Reference': prev_ref,
#                             'Alternative': prev_alt,
#                             'Genotype': prev_genotype,
#                             'Haploblock': prev_haploblock
#                         }
                    
#                     # Check for the next row
#                     if i < len(lines) - 1:
#                         next_cols = lines[i + 1].strip().split('\t')
#                         next_ref = next_cols[2]
#                         next_alt = next_cols[3]
#                         next_child_data = next_cols[child_column_index]
#                         next_genotype, next_haploblock = next_child_data.split(':') if ':' in next_child_data else (next_child_data, None)
#                         next_row = {
#                             'Chromosome': next_cols[0],
#                             'Position': int(next_cols[1]),
#                             'Reference': next_ref,
#                             'Alternative': next_alt,
#                             'Genotype': next_genotype,
#                             'Haploblock': next_haploblock
#                         }
                    
#                     found = True
#                     break
        
#         if found:
#             # Append all relevant rows to the data list
#             if previous_row:
#                 extracted_data.append(previous_row)
#             if target_row:
#                 extracted_data.append(target_row)
#             if next_row:
#                 extracted_data.append(next_row)
#             # Convert the data to a pandas DataFrame
#             dataframe = pd.DataFrame(extracted_data)
#             return dataframe, previous_row['Position'], target_row["Position"], next_row["Position"]

#         return None

import pandas as pd

def parse_vcf(file_name, chromosome, position, child_column_index):
    # Initialize data storage
    extracted_data = []
    target_row = None
    previous_row = None
    next_row = None

    # Open the file and process it line by line
    with open(file_name, 'r') as file:
        lines = file.readlines()

        # Filter out header lines and empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

        # Locate the target position
        for i, line in enumerate(data_lines):
            cols = line.split('\t')
            current_chrom = cols[0]
            current_pos = int(cols[1])

            if current_chrom == chromosome and current_pos == position:
                # Extract data for the target row
                ref = cols[2]
                alt = cols[3]
                child_column_data = cols[child_column_index]
                genotype, haploblock = child_column_data.split(':') if ':' in child_column_data else (child_column_data, None)
                target_row = {
                    'Chromosome': current_chrom,
                    'Position': current_pos,
                    'Reference': ref,
                    'Alternative': alt,
                    'Genotype': genotype,
                    'Haploblock': haploblock
                }

                # Search for the nearest previous row with a valid haploblock
                for j in range(i - 1, -1, -1):
                    prev_cols = data_lines[j].split('\t')
                    prev_haploblock = prev_cols[child_column_index].split(':')[1] if ':' in prev_cols[child_column_index] else None
                    if prev_haploblock not in [None, '.']:
                        prev_ref = prev_cols[2]
                        prev_alt = prev_cols[3]
                        prev_genotype, _ = prev_cols[child_column_index].split(':') if ':' in prev_cols[child_column_index] else (prev_cols[child_column_index], None)
                        previous_row = {
                            'Chromosome': prev_cols[0],
                            'Position': int(prev_cols[1]),
                            'Reference': prev_ref,
                            'Alternative': prev_alt,
                            'Genotype': prev_genotype,
                            'Haploblock': prev_haploblock
                        }
                        break

                # Search for the nearest next row with a valid haploblock
                for j in range(i + 1, len(data_lines)):
                    next_cols = data_lines[j].split('\t')
                    next_haploblock = next_cols[child_column_index].split(':')[1] if ':' in next_cols[child_column_index] else None
                    if next_haploblock not in [None, '.']:
                        next_ref = next_cols[2]
                        next_alt = next_cols[3]
                        next_genotype, _ = next_cols[child_column_index].split(':') if ':' in next_cols[child_column_index] else (next_cols[child_column_index], None)
                        next_row = {
                            'Chromosome': next_cols[0],
                            'Position': int(next_cols[1]),
                            'Reference': next_ref,
                            'Alternative': next_alt,
                            'Genotype': next_genotype,
                            'Haploblock': next_haploblock
                        }
                        break
                break

        # Append the valid rows to the data list
        if target_row:
            if previous_row:
                extracted_data.append(previous_row)
            extracted_data.append(target_row)
            if next_row:
                extracted_data.append(next_row)

        # Convert the data to a pandas DataFrame and return it
        if extracted_data:
            dataframe = pd.DataFrame(extracted_data)
            return dataframe, previous_row['Position'] if previous_row else None, target_row['Position'], next_row['Position'] if next_row else None

        return None

      
# def parse_sam_file(sam_file, positions_of_interest):
#     """
#     Parse the SAM file and extract allele and quality information for specific positions.
#     """
#     read_data = defaultdict(dict)  # Dictionary to store data by read name

#     # Open and parse the SAM file
#     with open(sam_file, "r") as f:
#         for line in f:
#             if line.startswith("@"):
#                 continue  # Skip header lines
#             fields = line.strip().split("\t")
#             read_name = fields[0]
#             pos = int(fields[3])  # 1-based position where alignment starts
#             cigar = fields[5]
#             seq = fields[9]
#             qual = fields[10]  # Read quality
#             mapping_quality = fields[4]  # Mapping quality
#             ps_tag = None
            
#             # Get PS tag if present
#             for field in fields[11:]:
#                 if field.startswith("PS:"):
#                     ps_tag = field.split(":")[2]

#             # Track alleles and qualities for this read
#             alleles = {}
#             qualities = {}
#             mapping_qualities = {}
#             read_index = 0
#             ref_position = pos  # Start of alignment

#             # Parse the CIGAR string
#             for match in re.finditer(r"(\d+)([MIDNSHP=X])", cigar):
#                 length, operation = int(match.group(1)), match.group(2)
#                 if operation in "M=X":  # Alignment match or mismatch
#                     for position in positions_of_interest:
#                         if ref_position <= position < ref_position + length:
#                             index_in_read = read_index + (position - ref_position)
#                             alleles[position] = seq[index_in_read]
#                             qualities[position] = qual[index_in_read]
#                     ref_position += length
#                     read_index += length
#                 elif operation == "I":  # Insertion
#                     read_index += length
#                 elif operation == "D":  # Deletion
#                     ref_position += length
#                 elif operation in "NSHP":  # Skipped regions, hard/soft clipping
#                     if operation in "SH":
#                         read_index += length
#                     else:
#                         ref_position += length
            
#             # Store read-level information
#             read_data[read_name]["alleles"] = read_data[read_name].get("alleles", {})
#             read_data[read_name]["qualities"] = read_data[read_name].get("qualities", {})
#             read_data[read_name]["alleles"].update(alleles)
#             read_data[read_name]["qualities"].update(qualities)
#             read_data[read_name]["mapping_quality"] = mapping_quality
#             read_data[read_name]["ps"] = ps_tag

#     return read_data




def parse_sam_file(sam_file, positions_of_interest):
    """
    Parse the SAM file and extract allele, quality, and position information for specific positions,
    while identifying paired-end reads.
    """
    read_data = defaultdict(lambda: {"read1": None, "read2": None})  # Group data by read name

    def parse_cigar(cigar, seq, qual, pos, positions_of_interest):
        """
        Parse the CIGAR string and extract alleles and qualities at specific positions.
        """
        alleles = {}
        qualities = {}
        read_index = 0
        ref_position = pos
        

        for match in re.finditer(r"(\d+)([MIDNSHP=X])", cigar):
            length, operation = int(match.group(1)), match.group(2)
            if operation in "M=X":  # Alignment match or mismatch
                for position in positions_of_interest:
                    if ref_position <= position < ref_position + length:
                        index_in_read = read_index + (position - ref_position)
                        alleles[position] = seq[index_in_read]
                        qualities[position] = qual[index_in_read]
                ref_position += length
                read_index += length
            elif operation == "I":  # Insertion
                read_index += length
            elif operation == "D":  # Deletion
                ref_position += length
            elif operation in "NSHP":  # Skipped regions, hard/soft clipping
                if operation in "SH":
                    read_index += length
                else:
                    ref_position += length
        return alleles, qualities

    with open(sam_file, "r") as f:
        for line in f:
            if line.startswith("@"):
                continue  # Skip header lines

            fields = line.strip().split("\t")
            read_name = fields[0]
            flag = int(fields[1])
            pos = int(fields[3])  # 1-based position where alignment starts
            cigar = fields[5]
            seq = fields[9]
            qual = fields[10]
            mapping_quality = fields[4]  # Mapping quality
            ps_tag = None

            # Get PS tag if present
            for field in fields[11:]:
                if field.startswith("PS:"):
                    ps_tag = field.split(":")[2]

            # Parse alleles and qualities at positions of interest
            alleles, qualities = parse_cigar(cigar, seq, qual, pos, positions_of_interest)

            # Determine if the read is the first or second in the pair
            read_info = {
                "start_position": pos,  # Starting position of the read
                "alleles": alleles,
                "qualities": qualities,
                "mapping_quality": mapping_quality,
                "ps": ps_tag,
            }

            if flag & 0x40:  # First in pair
                read_data[read_name]["read1"] = read_info
            elif flag & 0x80:  # Second in pair
                read_data[read_name]["read2"] = read_info

    return read_data


def analyze_allele_association(read_data, positions_of_interest):
    """
    Analyze allele associations between positions of interest and group by haploblock,
    considering paired-end reads (read1 and read2).
    """
    results = []

    for read_name, data in read_data.items():
        read1 = data.get("read1")
        read2 = data.get("read2")
                
        if read1 and read2:
            alleles1 = read1["alleles"]
            alleles2 = read2["alleles"]
            
            if positions_of_interest[0] in alleles1 and positions_of_interest[1] in alleles2:
                results.append({
                    "Read Name": f"{read_name}_paired",
                    "Allele 1": alleles1[positions_of_interest[0]],
                    "Allele 2": alleles2[positions_of_interest[1]],
                    "PS": read1["ps"] or read2["ps"],  # Use PS from either read if available
                    "Mapping Quality": min(read1["mapping_quality"], read2["mapping_quality"]),  # Combine mapping qualities
                    "Start Position": (read1["start_position"], read2["start_position"])  # Tuple of start positions
                })
            elif positions_of_interest[1] in alleles1 and positions_of_interest[0] in alleles2:
                results.append({
                    "Read Name": f"{read_name}_paired",
                    "Allele 1": alleles1[positions_of_interest[1]],
                    "Allele 2": alleles2[positions_of_interest[0]],
                    "PS": read1["ps"] or read2["ps"],  # Use PS from either read if available
                    "Mapping Quality": min(read1["mapping_quality"], read2["mapping_quality"]),  # Combine mapping qualities
                    "Start Position": (read2["start_position"], read1["start_position"])  # Tuple of start positions
                })
                
    if results:
        # Create a DataFrame
        df = pd.DataFrame(results)

        # Rename columns dynamically based on positions
        df = df.rename(columns={
            "Allele 1": str(positions_of_interest[0]),
            "Allele 2": str(positions_of_interest[1])
        })
        # Group by alleles and PS, then count occurrences
        grouped_counts = df.groupby([str(positions_of_interest[0]), str(positions_of_interest[1]), 'PS']).size().reset_index(name='Count')

        # Find the maximum count per haploblock
        max_counts_per_haploblock = grouped_counts.loc[grouped_counts.groupby("PS")["Count"].idxmax()]
        return df, grouped_counts, max_counts_per_haploblock


def analyze_haploblocks(df_positions, df_haploblock, position_1, position_2):
    """
    Analyze haploblock allele consistency between two positions.
    
    Args:
        df_positions (pd.DataFrame): DataFrame with position, reference, and genotype data.
        df_haploblock (pd.DataFrame): DataFrame with allele associations for the haploblock.
        position_1 (int): The first position to analyze.
        position_2 (int): The second position to analyze.

    Returns:
        str: Result of the analysis for the given haploblock.
    """
    # Extract haploblock information
    df_positions = df_positions.copy()
    # print("DF haploblock", df_haploblock.head(2))
    pos_1_allele = df_haploblock[str(position_1)].iloc[0]
    pos_2_allele = df_haploblock[str(position_2)].iloc[0]
    

    # Extract reference and genotype data
    reference_1 = df_positions.loc[df_positions["Position"] == position_1, "Reference"].iloc[0]
    reference_2 = df_positions.loc[df_positions["Position"] == position_2, "Reference"].iloc[0]
    genotype_1 = df_positions.loc[df_positions["Position"] == position_1, "Genotype"].iloc[0]
    genotype_2 = df_positions.loc[df_positions["Position"] == position_2, "Genotype"].iloc[0]

    # Check which one is already assigned :
    if "|" in genotype_1 :
        if reference_1 == pos_1_allele and pos_2_allele == reference_2:
            df_positions.loc[df_positions["Position"] == position_2, "Genotype"] = genotype_1            
        elif reference_1 != pos_1_allele and pos_2_allele == reference_2:
            df_positions.loc[df_positions["Position"] == position_2, "Genotype"] = genotype_1[::-1]
        elif reference_1 == pos_1_allele and pos_2_allele != reference_2:
            df_positions.loc[df_positions["Position"] == position_2, "Genotype"] = genotype_1[::-1]
        elif reference_1 != pos_1_allele and pos_2_allele != reference_2:
            df_positions.loc[df_positions["Position"] == position_2, "Genotype"] = genotype_1
        df_positions.loc[df_positions["Position"] == position_2, "Haploblock"] = df_positions.loc[df_positions["Position"] == position_1, "Haploblock"].values[0]
        
    elif "|" in genotype_2:
        if reference_1 == pos_1_allele and pos_2_allele == reference_2:
            df_positions.loc[df_positions["Position"] == position_1, "Genotype"] = genotype_2
        elif reference_1 != pos_1_allele and pos_2_allele == reference_2:
            df_positions.loc[df_positions["Position"] == position_1, "Genotype"] = genotype_2[::-1]
        elif reference_1 == pos_1_allele and pos_2_allele != reference_2:
            df_positions.loc[df_positions["Position"] == position_1, "Genotype"] = genotype_2[::-1]
        elif reference_1 != pos_1_allele and pos_2_allele != reference_2:
            df_positions.loc[df_positions["Position"] == position_1, "Genotype"] = genotype_2
        df_positions.loc[df_positions["Position"] == position_1, "Haploblock"] = df_positions.loc[df_positions["Position"] == position_2, "Haploblock"].values[0]
        

    return df_positions
        
    # else:
   
def compare_genotypes(df_prev, df_post, dnm_pos):
    ## Given the genotype datafrfame of the positions with the prev, next positions
    ## Check if they have the same genotypes, if yes, return the genotypes
    ## If not, return None
    if df_prev.loc[df_prev["Position"] == dnm_pos, "Genotype"].values[0] == df_post.loc[df_post["Position"] == dnm_pos, "Genotype"].values[0] :
        return df_prev.loc[df_prev["Position"] == dnm_pos, "Genotype"].values[0], df_prev.loc[df_prev["Position"] == dnm_pos, "Haploblock"].values[0]
    return None

if __name__ == "__main__":
    
    # Only take reads that have good mapping quality
    mapping_quality_threshold = 30
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse VCF-like file and extract specific information.")
    parser.add_argument("--sam_file", type=str, required=True, help="samfile")
    parser.add_argument("--vcf_file", type=str, required=True, help="Input file name")
    parser.add_argument("--chromosome", type=str, required=True, help="Chromosome to filter")
    parser.add_argument("--position", type=int, required=True, help="Position to filter")
    parser.add_argument("--child", type=int, required=True, help="Column index for child (1-based index)")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write outputs (modified output of whasthap)")
    parser.add_argument("--output_filename", type=str, required=True, help="Name of the output file")
    args = parser.parse_args()
    sam_file = args.sam_file
    
    
    
    # Adjust column index (convert 1-based to 0-based)
    child_column_index = args.child - 1
    
    # Parse the file
    vcf_contents = parse_vcf(args.vcf_file, args.chromosome, args.position, child_column_index)
    if vcf_contents is not None:
        
        df, prev_pos, dnm_pos, next_pos = vcf_contents
        print("prev position:", prev_pos)
        print("post position:", next_pos)
        
        if prev_pos is not None and next_pos is not None:
            # print(df) ## Full vcf file with columns like REF and ALT

            # Parse the SAM file
            read_data = parse_sam_file(sam_file, [prev_pos, dnm_pos] )
            # print("Number of reads", len(read_data))
            
            # Filter reads by mapping quality
            filtered_read_data = {}
            for read_name, data in read_data.items():
                if data["read1"] and data["read2"]:
                    if (int(data["read2"]["mapping_quality"]) >= mapping_quality_threshold and \
                        int(data["read1"]["mapping_quality"]) >= mapping_quality_threshold ):
                        filtered_read_data[read_name] = data        
            read_data = filtered_read_data
            
            # Analyze allele associations
            prev_allele_association = analyze_allele_association(read_data, [prev_pos, dnm_pos])
            if prev_allele_association is not None:
                read_alleles_prev, grouped_counts_prev, max_counts_per_haploblock_prev = prev_allele_association
            # print(read_alleles_prev)
            # Display results
            # print("Grouped Counts by Haploblock:\n", grouped_counts_prev)
            # print("\nMaximum Counts Per Haploblock:\n", max_counts_per_haploblock_prev)
            
            # Parse the SAM file
            read_data = parse_sam_file(sam_file, [dnm_pos, next_pos])
                
            # Filter reads by mapping quality
            filtered_read_data = {}
            for read_name, data in read_data.items():
                if data["read1"] and data["read2"]:
                    if (int(data["read2"]["mapping_quality"]) >= mapping_quality_threshold and \
                        int(data["read1"]["mapping_quality"]) >= mapping_quality_threshold ):
                        filtered_read_data[read_name] = data        
            read_data = filtered_read_data

            # Analyze allele associations
            next_allele_association = analyze_allele_association(read_data, [dnm_pos, next_pos])
            if next_allele_association is not None:
                read_alleles_post, grouped_counts_post, max_counts_per_haploblock_post = next_allele_association

            # Display results
            
            # print("Grouped Counts by Haploblock:\n", grouped_counts_post)
            # print("\nMaximum Counts Per Haploblock:\n", max_counts_per_haploblock_post)
            
            
            ## These will be inputs for the function compare_genotypes
            if next_allele_association is not None and prev_allele_association is not None:
                # Analyze haploblock 1
                result1 = analyze_haploblocks(df_positions=df, df_haploblock=max_counts_per_haploblock_prev, position_1=prev_pos, position_2=dnm_pos)
                # print(result1)

                # Analyze haploblock 2
                result2 = analyze_haploblocks(df_positions=df, df_haploblock=max_counts_per_haploblock_post, position_1=dnm_pos, position_2=next_pos)
                # print(result2)

                ## Check consistency of genotypes in both
                genotype_dnm = compare_genotypes(df_prev=result1, df_post=result2, dnm_pos=args.position)
                if genotype_dnm is not None:         
                    original_df = pd.read_csv(args.vcf_file, sep="\t", header=0)
                    original_df.columns = ['#CHROM', 'POS', 'Reference', 'Alternative', \
                        'Child', 'Father', 'Mother']
                    print("Found haploblock for the child")
                    original_df.loc[(original_df["POS"] == args.position) & (original_df["#CHROM"] == args.chromosome), "Child"] = genotype_dnm[0] + ":" + genotype_dnm[1]  
                    print(original_df.loc[(original_df["POS"] == args.position) & (original_df["#CHROM"] == args.chromosome)])
                    # Select some columns
                    original_df = original_df[['#CHROM', 'POS', 'Child', 'Father', 'Mother']]
                    # Write out results over the original_df
                    original_df.to_csv(args.out_dir + "/" + args.output_filename, sep="\t", index=False)
            else:
                print(f"No read covering 3 positions {prev_pos, dnm_pos, next_pos}")