__kernel void matmul(
    __global float* result,
    __global float* A,
    __global float* X,
    int n_cols)
{
    int tx = get_global_id(0);

    float value = 0;
    for (unsigned int i = 0; i < n_cols; ++i) {
        value += A[tx * n_cols + i] * X[i];
    }
    result[tx] = value;
}
