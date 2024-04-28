def print_indices_sequential(n):
    tile_size = 2  # Assuming the tile size is fixed to 2x2
    num_blocks = n // tile_size  # Calculate the number of blocks per dimension

    # Iterate over each block in the grid
    for by in range(num_blocks):
        for bx in range(num_blocks):
            # Each thread within a block
            for ty in range(tile_size):
                for tx in range(tile_size):
                    # Calculate the row and column in the global matrix for each thread
                    row = by * tile_size + ty
                    col = bx * tile_size + tx

                    print(f"Block: ({bx},{by}), Thread: ({tx},{ty})")
                    print(f"tx: {tx}, ty: {ty}, bx: {bx}, by: {by}")
                    print(f"row: {row}, col: {col}")

                    for i in range(n // tile_size):
                        # Calculate index for A
                        global_row = row * n  # The starting index of the row
                        column_set = i * tile_size + tx
                        mem_index_a = global_row + column_set

                        # Calculate index for B
                        row_set = i * tile_size  # The starting row for the current tile of B
                        global_col = col  # Column remains the same across different tile sets for B
                        mem_index_b = (row_set + ty) * n + global_col

                        print(f"Iteration: {i}")
                        print(f"Index calculation for A (row start: {global_row}, column set: {column_set}, index: {mem_index_a})")
                        print(f"Index calculation for B (row set: {row_set}, row within set: {ty}, index: {mem_index_b})")

# Example usage
print_indices_sequential(4)  # Example matrix size N=4
