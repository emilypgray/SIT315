__kernel void add_matrix(const int rows, const int cols,
                      const __global int* A, 
                      const __global int* B, 
                      __global int* C) {

    // get the global ids of the thread
    const int i = get_global_id(0);    
    const int j = get_global_id(1);
 
    int sum = 0;

    if (i < rows && j < cols)
    {
        // loop through one row of A and one column of B
        // and multiply each element. Add product to sum variable
        for (int m = 0; m < cols; m++)
        {
            sum += A[i * cols + m] * B[j + m * cols];
        }
        // assign C the total value of sum
        C[i * cols + j] = sum;
    }
}
