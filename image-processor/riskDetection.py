import numpy as np

def split_matrix(matrix, sections_shape):
    rows, cols = matrix.shape
    sec_rows, sec_cols = sections_shape

    # Calculate the size of each submatrix by dividing the cropped dimensions by the section counts
    submatrix_rows = rows // sec_rows
    submatrix_cols = cols // sec_cols
    
    # Crop the matrix to be exactly divisible by the sections_shape
    cropped_matrix = matrix[:submatrix_rows * sec_rows, :submatrix_cols * sec_cols]
    
    # Split the cropped matrix into the specified number of sections
    split_matrices = [
        cropped_matrix[i * submatrix_rows:(i + 1) * submatrix_rows, 
                       j * submatrix_cols:(j + 1) * submatrix_cols]
        for i in range(sec_rows) for j in range(sec_cols)
    ]
    
    return split_matrices

def getRiskMatrix (disparity, calibration):
    sections = split_matrix(disparity["map"], (3, 3))
    
    result = []
    for section in sections:
        result.append(np.mean(section))
    result = np.array(result).astype(int)
        
    print(result)
    
    return [0,0,0,0,0,0,0,0,0]

# def create_display_matrix(source_array, section_shape, scale):
#     tRow = section_shape[0] * scale
#     tCol = section_shape[1] * scale
    
#     target_matrix = np.zeros((tRow, tCol), dtype=int)
    
#     for i in range(tRow):
#         for j in range(tCol):
#             value = source_array[]
#             target_matrix[i * section_shape[0]:(i + 1) * section_shape[0],
#                            j * section_shape[1]:(j + 1) * section_shape[1]] = value
    
#     return target_matrix

def create_display_matrix(matrix, target_shape):
    # Calculate the scale factor for rows and columns
    row_scale = target_shape[0] // matrix.shape[0]
    col_scale = target_shape[1] // matrix.shape[1]
    
    # Repeat the rows and columns
    scaled_matrix = np.repeat(np.repeat(matrix, row_scale, axis=0), col_scale, axis=1)
    
    return scaled_matrix