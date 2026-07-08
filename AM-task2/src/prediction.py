

if __name__ == "__main__":
  print("prediction")

  # read commands/[file]

  # get script executed:
    # - spmv (SParse Matrix–Vector multiplication)
    # - mpirun

  # get parameters: tensor, matrix, iterations

  # if mpi:
    # get number of process

  # if matrix:
    # get size of the matrix (rows, cols, nnz)

    # estimate GPU memory
      # values      = nnz × 8 bytes
      # col_indices = nnz × 4 bytes
      # row_ptr     = (rows + 1) × 4 bytes
      # so => matrix_memory = nnz × (8 + 4) + (rows + 1) × 4

  # if vector:
    # get size of the vector (rows)

    # x vector = rows × 8
    # y vector = rows × 8
    # so => vector_memory = 2 × rows × 8



