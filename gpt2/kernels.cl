#define ELEMENTWISE(i, n, out, expr) ({ \
  int i = get_global_id(0); \
  if (i < (n)) { out = (expr); } \
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

__kernel void norm(__global float* result, __global float* weight, __global float* bias, __global float* x, int offset) {
  int tx = get_global_id(0);
  int size = get_global_size(0);
  float mean = 0, var = 0;
  for (int i = 0; i < size; i++) {
    mean += x[i] / size;
  }
  for (int i = 0; i < size; i++) {
    var += (x[i] - mean) * (x[i] - mean) / size;
  }
  float std = sqrt(var) + 1e-5; // epsilon
  float y = (x[tx] - mean) / std;
  result[tx] = y * weight[offset + tx] + bias[offset + tx];
}
