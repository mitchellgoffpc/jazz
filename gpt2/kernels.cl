#define ELEMENTWISE(i, n, out, expr) ({ \
  int i = get_global_id(0); \
  if (i < (n)) { out = (expr); } \
})

#define SUM(lid, local_size, local_data) ({ \
  for (int i = local_size/2; i > 0; i >>= 1) { \
    if (lid < i) \
      local_data[lid] += local_data[lid + i]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
  } \
})

__kernel void copy(__global float* dst, __global float* src, int dst_offset, int src_offset, int dst_stride, int src_stride) {
  int col = get_local_id(0);
  int row = get_global_id(0) / get_local_size(0);
  dst[dst_offset + row*dst_stride + col] = src[src_offset + row*src_stride + col];
}

__kernel void add(__global float* result, __global float* x, __global float* y, int x_offset, int y_offset, int n) {
  ELEMENTWISE(i, n, result[i], x[x_offset + i] + y[y_offset + i]);
}
__kernel void scale( __global float* x, float c, int n) {
  ELEMENTWISE(i, n, x[i], x[i] * c);
}
__kernel void gelu(__global float* x, int n) {
  ELEMENTWISE(i, n, x[i], ({
    float y = x[i];
    0.5 * y * (1 + tanh(sqrt(2 / M_PI) * (y + 0.044715 * y*y*y)));
  }));
}

__kernel void matmul(__global float* result, __global float* A, __global float* x, int n_aisles, int n_rows, int n_cols, int A_offset, int x_offset, int A_stride) {
  int gid = get_global_id(0);
  int row = gid % n_rows;
  int aisle = gid / n_rows;
  float value = 0;
  for (int col = 0; col < n_cols; col++) {
    value += A[A_offset + (aisle * A_stride) + (row * n_cols) + col] * x[x_offset + (aisle * n_cols) + col];
  }
  if (gid < n_aisles * n_rows) {
    result[aisle * n_rows + row] = value;
  }
}
__kernel void matvmul(__global float* result, __global float* x, __global float* A, int n_aisles, int n_rows, int n_cols, int x_offset, int A_offset, int A_stride) {
  int gid = get_global_id(0);
  int col = gid % n_cols;
  int aisle = gid / n_cols;
  float value = 0;
  for (int row = 0; row < n_rows; row++) {
    value += A[A_offset + (aisle * A_stride) + (row * n_cols) + col] * x[x_offset + (aisle * n_rows) + row];
  }
  if (gid < n_aisles * n_cols) {
    result[aisle * n_cols + col] = value;
  }
}

__kernel void embedding(__global float* result, __global float* weights, int idx, int embed_size) {
    int tx = get_global_id(0);
    result[tx] = weights[idx * embed_size + tx];
}

__kernel void softmax(__global float* x, int n_rows, int n_cols) {
  int gid = get_global_id(0);
  int row = gid / n_cols;
  int offset = row * n_cols;

  // NOTE: This probably isn't safe without a global memory fence, we should just use a separate output buffer
  float max = 0;
  for (int i = 0; i < n_cols; i++) {
    max = x[offset+i] > max ? x[offset+i] : max;
  }

  float sum = 0;
  for (int i = 0; i < n_cols; i++) {
    sum += exp(x[offset+i] - max);
  }
  if (gid < n_rows * n_cols) {
    x[gid] = exp(x[gid] - max) / sum;
  }
}

__kernel void norm_a(__global float* output, __global float* input, __local float* local_data) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int local_size = get_local_size(0);
  int global_size = get_global_size(0);
  int num_blocks = global_size / local_size;

  // Load the data, compute partial sums, and store
  local_data[lid] = input[gid];
  barrier(CLK_LOCAL_MEM_FENCE);
  SUM(lid, local_size, local_data);
  if (lid == 0)
    output[gid / local_size] = local_data[0] / local_size;
}

__kernel void norm_b(__global float* output, __global float* input, __local float* local_data) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int local_size = get_local_size(0);
  int global_size = get_global_size(0);
  int num_blocks = global_size / local_size;

  // Load the partial sums from all blocks and compute the final mean
  local_data[lid] = lid < num_blocks ? output[lid] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  SUM(lid, local_size, local_data);
  float mean = local_data[0] / num_blocks;

  // Load the data, compute partial sums, and store
  float dist = input[gid] - mean;
  local_data[lid] = dist * dist;
  barrier(CLK_LOCAL_MEM_FENCE);
  SUM(lid, local_size, local_data);
  if (lid == 0)
    output[num_blocks + gid / local_size] = local_data[0] / local_size;
}

__kernel void norm_c(__global float* output, __global float* partials, __global float* weight, __global float* bias, __global float* input, __local float* local_data, int offset) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int local_size = get_local_size(0);
  int global_size = get_global_size(0);
  int num_blocks = global_size / local_size;

  // Load the partial sums from all blocks and compute the final mean
  local_data[lid] = lid < num_blocks ? partials[lid] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  SUM(lid, local_size, local_data);
  float mean = local_data[0] / num_blocks;

  // Load the partial sums from all blocks and compute the final variance
  local_data[lid] = lid < num_blocks ? partials[num_blocks + lid] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  SUM(lid, local_size, local_data);
  float var = local_data[0] / num_blocks;
  float std = sqrt(var) + 1e-5; // epsilon

  // Normalize, apply gamma/beta, and store
  float y = (input[gid] - mean) / std;
  output[gid] = y * weight[offset + gid] + bias[offset + gid];
}
