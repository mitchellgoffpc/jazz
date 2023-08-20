__kernel void add(__global float* result, __global float* x, __global float* y) {
  int tx = get_global_id(0);
  result[tx] = x[tx] + y[tx];
}

__kernel void matmul(__global float* result, __global float* A, __global float* X, int n_cols) {
  int tx = get_global_id(0);
  float value = 0;
  for (int i = 0; i < n_cols; i++) {
    value += A[tx * n_cols + i] * X[i];
  }
  result[tx] = value;
}

__kernel void embedding(__global float* result, __global float* weights, int idx, int embed_size) {
    int tx = get_global_id(0);
    result[tx] = weights[idx * embed_size + tx];
}
