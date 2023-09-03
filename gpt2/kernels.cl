#define GLOBALS \
  int global_idx = get_global_id(0); \
  int thread_idx = get_local_id(0); \
  int block_dim = get_local_size(0); \
  int global_dim = get_global_size(0); \
  int block_idx = global_idx / block_dim; \
  int grid_dim = global_dim / block_dim

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
  local_data[0]; \
})

// This only seems to work without barriers when local_size <= 32
// #define SUM(thread_idx, local_size, local_data) ({ \
//   if (thread_idx < 16) local_data[thread_idx] += local_data[thread_idx + 16]; \
//   if (thread_idx < 8) local_data[thread_idx] += local_data[thread_idx + 8]; \
//   if (thread_idx < 4) local_data[thread_idx] += local_data[thread_idx + 4]; \
//   if (thread_idx < 2) local_data[thread_idx] += local_data[thread_idx + 2]; \
//   if (thread_idx < 1) local_data[thread_idx] += local_data[thread_idx + 1]; \
//   local_data[0]; \
// })

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

__kernel void matmul_a(__global float* result, __global float* A, __global float* x, __local float* local_data,
                       int n_aisles, int n_rows, int n_cols, int A_offset, int x_offset, int A_stride) {
  GLOBALS;
  int row = (global_idx / n_cols) % n_rows;
  int aisle = (global_idx / n_cols) / n_rows;
  local_data[thread_idx] = A[A_offset + (aisle * A_stride) + (row * n_cols) + (global_idx % n_cols)] * x[x_offset + (aisle * n_cols) + (global_idx % n_cols)];
  barrier(CLK_LOCAL_MEM_FENCE);

  float sum = SUM(thread_idx, block_dim, local_data);
  if (thread_idx == 0)
    result[block_idx] = sum;
}

__kernel void matmul_b(__global float* result, __global float* partials, __local float* local_data, int chunks_per_row) {
  GLOBALS;
  int row = global_idx / chunks_per_row;
  local_data[thread_idx] = thread_idx < chunks_per_row ? partials[row * chunks_per_row + thread_idx] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  float sum = SUM(thread_idx, block_dim, local_data);
  if (thread_idx == 0)
    result[block_idx] = sum;
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
  GLOBALS;

  // Load the data and compute partial sums for mean
  local_data[thread_idx] = input[global_idx];
  barrier(CLK_LOCAL_MEM_FENCE);

  float sum = SUM(thread_idx, block_dim, local_data);
  if (thread_idx == 0)
    output[block_idx] = sum / block_dim;
}

__kernel void norm_b(__global float* output, __global float* input, __local float* local_data) {
  GLOBALS;

  // Load the partial sums from all blocks and compute the final mean
  local_data[thread_idx] = thread_idx < grid_dim ? output[thread_idx] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  float mean = SUM(thread_idx, block_dim, local_data) / grid_dim;

  // Load the data and compute partial sums for variance
  float dist = input[global_idx] - mean;
  local_data[thread_idx] = dist * dist;
  barrier(CLK_LOCAL_MEM_FENCE);

  float sum = SUM(thread_idx, block_dim, local_data);
  if (thread_idx == 0) {
    output[grid_dim + block_idx] = sum / block_dim;
    output[grid_dim * 2] = mean;
  }
}

__kernel void norm_c(__global float* output, __global float* partials, __global float* weight, __global float* bias, __global float* input, __local float* local_data, int offset) {
  GLOBALS;

  // Load the partial sums from all blocks and compute the final variance
  local_data[thread_idx] = thread_idx < grid_dim ? partials[grid_dim + thread_idx] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  float var = SUM(thread_idx, block_dim, local_data) / grid_dim;
  float std = sqrt(var) + 1e-5; // epsilon
  float mean = partials[grid_dim * 2];

  // Normalize, apply gamma/beta, and store
  float y = (input[global_idx] - mean) / std;
  output[global_idx] = y * weight[offset + global_idx] + bias[offset + global_idx];
}
