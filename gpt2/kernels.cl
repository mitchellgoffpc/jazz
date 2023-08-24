__kernel void copy(__global float* dst, __global float* src, int dst_offset, int src_offset) {
  int tx = get_global_id(0);
  dst[dst_offset + tx] = src[src_offset + tx];
}

__kernel void add(__global float* result, __global float* x, __global float* y) {
  int tx = get_global_id(0);
  result[tx] = x[tx] + y[tx];
}

__kernel void scale( __global float* x, float c) {
  int tx = get_global_id(0);
  x[tx] = x[tx] * c;
}

__kernel void gelu(__global float* x) {
  int tx = get_global_id(0);
  float y = x[tx];
  x[tx] = 0.5 * y * (1 + tanh(sqrt(2 / M_PI) * (y + 0.044715 * y*y*y)));
}

__kernel void matmul(__global float* result, __global float* A, __global float* x, int A_offset, int x_offset, int n_cols) {
  int tx = get_global_id(0);
  float value = 0;
  for (int i = 0; i < n_cols; i++) {
    value += A[A_offset, tx * n_cols + i] * x[x_offset + i];
  }
  result[tx] = value;
}

__kernel void embedding(__global float* result, __global float* weights, int idx, int embed_size) {
    int tx = get_global_id(0);
    result[tx] = weights[idx * embed_size + tx];
}

__kernel void norm(__global float* result, __global float* weight, __global float* bias, __global float* x, int size, int layer) {
  int tx = get_global_id(0);
  float mean = 0, var = 0;
  for (int i = 0; i < size; i++) {
    mean += x[i] / size;
  }
  for (int i = 0; i < size; i++) {
    var += (x[i] - mean) * (x[i] - mean) / size;
  }
  float std = sqrt(var) + 1e-5; // epsilon
  float y = (x[tx] - mean) / std;
  result[tx] = y * weight[layer * size + tx] + bias[layer * size + tx];
}
