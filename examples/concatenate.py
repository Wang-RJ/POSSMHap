
import numpy as np
import itertools

# Sample matrices M1, M2, and M3
# M1 = np.array([[1, 1], [0, 1], [0, 1]])  # Matrix 1
# M2 = np.array([[0, 0], [1, 0]])  # Matrix 2
# M3 = np.array([[0, 1], [1, 1], [1, 1]])  # Matrix 3

# Define M_star (one of the matrices in the list)
# M_star = M3  # Example: M_star is M2

# # List of matrices
# matrices = [M1, M_star, M3]

# Function to generate all super matrices with correct row-wise concatenation
def generate_super_matrices(matrices, M_star):
    
    # Remove empty matrices
    matrices = [matrix for matrix in matrices if len(matrix) > 0]
    if not matrices:
        return []
    
    if len(matrices) == 1:
        return matrices
    
    n = len(matrices)
    # List to hold all super matrices
    super_matrices = []

    print("Number of matrices to combine : ", n)
    # if n > 20:
    #     print("Too many combinations to consider")
    #     return super_matrices

    # Generate all combinations (binary choices of columns) except for M_star
    if M_star is not None:
        combinations = itertools.product([0, 1], repeat=n-1)  # Exclude M_star from permutations
        for combination in combinations:
            # Start with an empty list to hold rows for the current super matrix
            selected_rows = []
            other_rows = []
            
            for i, choice in enumerate(combination):
                if matrices[i] is M_star:
                    # If this matrix is M_star, keep both columns as they are
                    selected_rows += list(M_star[:, 0])
                    other_rows += list(M_star[:, 1])
                else:
                    # Append the selected column of the current matrix to the rows
                    selected_rows += list(matrices[i][:, choice])
                    # Append the other column of the current matrix to the rows
                    other_rows += list(matrices[i][:, 1 - choice])
            
            # Add M_star columns without changing their order
            if matrices[n-1] is M_star:
                selected_rows += list(M_star[:, 0])
                other_rows += list(M_star[:, 1])
            else:
                # Handle the last matrix in the permutation
                selected_rows += list(matrices[n-1][:, combination[-1]])
                other_rows += list(matrices[n-1][:, 1 - combination[-1]])
            
            # Concatenate all selected columns vertically (row-wise)
            super_matrix_left = np.vstack(selected_rows)
            super_matrix_right = np.vstack(other_rows)
            super_matrix = np.hstack((super_matrix_left, super_matrix_right))
            # in_super_matrix  = [np.array_equal(super_matrix, m) for m in super_matrices]
            # if len(in_super_matrix) == 0 or not any(in_super_matrix):
            #     super_matrices.append(super_matrix)
            super_matrices.append(super_matrix)
    print("The combined matrices are : ", super_matrices)
    return super_matrices

# # Generate all super matrices
# # all_super_matrices = generate_super_matrices(matrices, M_star)

# # # Print the result
# # for i, super_matrix in enumerate(all_super_matrices, 1):
# #     print(f"Super Matrix {i}:\n{super_matrix}\n")

# --- New code --- using HASH table

## -------------------------------------------------------------------------------------------
## using hash
# import numpy as np
# import itertools
# import time

# def generate_super_matrices(matrices, M_star):
    
#     # Remove empty matrices
#     matrices = [matrix for matrix in matrices if matrix.size > 0]
#     if not matrices:
#         return []
    
#     n = len(matrices)
#     if n == 1:
#         return matrices
    
#     print("Number of matrices to combine:", n)
#     if n > 20:
#         print("Too many combinations to consider")
#         return []
    
#     # Initialize list and set for super matrices and their hashes
#     super_matrices = []
#     super_matrices_hashes = set()
    
#     # Generate all combinations of column selections
#     combinations = itertools.product([0, 1], repeat=n - 1)
#     for combination in combinations:
#         selected_columns = []
#         other_columns = []
        
#         # Iterate over all matrices except the last one
#         for i in range(n - 1):
#             matrix = matrices[i]
#             if matrix is M_star:
#                 selected_columns.append(matrix[:, 0])
#                 other_columns.append(matrix[:, 1])
#             else:
#                 selected_columns.append(matrix[:, combination[i]])
#                 other_columns.append(matrix[:, 1 - combination[i]])
        
#         # Handle the last matrix
#         matrix = matrices[n - 1]
#         if matrix is M_star:
#             selected_columns.append(matrix[:, 0])
#             other_columns.append(matrix[:, 1])
#         else:
#             selected_columns.append(matrix[:, combination[-1]])
#             other_columns.append(matrix[:, 1 - combination[-1]])
        
#         # Concatenate the selected columns vertically
#         super_matrix_left = np.concatenate(selected_columns)
#         super_matrix_right = np.concatenate(other_columns)
#         super_matrix = np.column_stack((super_matrix_left, super_matrix_right))
        
#         # Use hash for quick duplicate checking
#         super_matrix_hash = hash(super_matrix.tobytes())
#         if super_matrix_hash not in super_matrices_hashes:
#             super_matrices.append(super_matrix)
#             super_matrices_hashes.add(super_matrix_hash)
    
#     return super_matrices

# Generate 16 sample matrices
## -------------------TEST ---------------------------------------------
# def generate_sample_matrices(num_matrices, num_rows=2):
#     matrices = []
#     for i in range(num_matrices):
#         # Create a matrix with 'num_rows' rows and 2 columns filled with random 0s and 1s
#         matrix = np.random.randint(0, 2, size=(num_rows, 2))
#         matrices.append(matrix)
#     return matrices

# # Set up the test case
# num_matrices = 16
# matrices = generate_sample_matrices(num_matrices)
# M_star = matrices[0]  # Let's assume the first matrix is M_star

# # Measure the runtime
# start_time = time.time()
# all_super_matrices = generate_super_matrices(matrices, M_star)
# end_time = time.time()

# # Print the runtime
# print(f"\nTotal super matrices generated: {len(all_super_matrices)}")
# print(f"Runtime for generating super matrices with {num_matrices} matrices: {end_time - start_time:.4f} seconds")
# print(all_super_matrices)



