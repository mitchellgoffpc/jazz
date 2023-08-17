#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define VECTOR_SIZE 1024

#define CL_CHECK(_expr) assert((_expr) == CL_SUCCESS);
#define CL_CHECK_ERR(_expr) ({       \
  cl_int err = CL_INVALID_VALUE;     \
  __typeof__(_expr) _ret = _expr;    \
  assert(_ret && err == CL_SUCCESS); \
  _ret;                              \
})


// Definitions

typedef struct {
  size_t num_layers;
  size_t num_heads;
  size_t embed_size;
  size_t vocab_size;
  size_t context_size;
} Config;

typedef struct {
  cl_mem weight;
  cl_mem bias;
  int size;
} LayerNorm;

typedef struct {
  cl_mem weight;
  cl_mem bias;
  int in_size;
  int out_size;
} Linear;

typedef struct {
  Config config;
  cl_mem embed_tokens;
  cl_mem embed_pos;
  LayerNorm ln1;
  LayerNorm ln2;
  Linear qkv;
  Linear proj;
  Linear fc1;
  Linear fc2;
  LayerNorm ln_out;
  cl_mem fc_out;
} GPT2;

typedef struct {
  cl_mem past;
  cl_mem x;
  cl_mem ln1;
  cl_mem qkv;
  cl_mem attn;
  cl_mem attn_out;
  cl_mem proj;
  cl_mem ln2;
  cl_mem fc1;
  cl_mem fc2;
  cl_mem out;
} State;

typedef struct {
  cl_platform_id* platforms;
  cl_device_id* devices;
  cl_context context;
  cl_command_queue command_queue;
} CL;


// Initialization

void init_cl(CL* cl) {
  cl_uint num_platforms, num_devices;

  CL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
  cl->platforms = malloc(num_platforms * sizeof(cl_platform_id));
  CL_CHECK(clGetPlatformIDs(num_platforms, cl->platforms, NULL));

  CL_CHECK(clGetDeviceIDs(cl->platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices));
  cl->devices = malloc(num_devices * sizeof(cl_device_id));
  CL_CHECK(clGetDeviceIDs(cl->platforms[0], CL_DEVICE_TYPE_GPU, num_devices, cl->devices, NULL));

  cl->context = CL_CHECK_ERR(clCreateContext(NULL, num_devices, cl->devices, NULL, NULL, &err));
  cl->command_queue = CL_CHECK_ERR(clCreateCommandQueue(cl->context, cl->devices[0], 0, &err));
}

cl_mem init_cl_buffer(CL* cl, int mode, float* buf, size_t size) {
  cl_mem cl_buf = CL_CHECK_ERR(clCreateBuffer(cl->context, mode, size * sizeof(float), NULL, &err));
  if (buf) CL_CHECK(clEnqueueWriteBuffer(cl->command_queue, cl_buf, CL_TRUE, 0, size * sizeof(float), buf, 0, NULL, NULL));
  return cl_buf;
}


// Cleanup

void free_cl(CL* cl) {
  CL_CHECK(clReleaseCommandQueue(cl->command_queue));
  CL_CHECK(clReleaseContext(cl->context));
  free(cl->platforms);
  free(cl->devices);
}


const char *saxpy_kernel =
"__kernel                                    \n"
"void saxpy_kernel(float alpha,              \n"
"                  __global float* A,        \n"
"                  __global float* B,        \n"
"                  __global float* C)        \n"
"{                                           \n"
"    int index = get_global_id(0);           \n"
"    C[index] = alpha * A[index] + B[index]; \n"
"}                                           \n";


int main() {
  // Create vectors A, B and C
  float alpha = 2.0;
  float* A = malloc(VECTOR_SIZE * sizeof(float));
  float* B = malloc(VECTOR_SIZE * sizeof(float));
  float* C = malloc(VECTOR_SIZE * sizeof(float));
  for (int i = 0; i < VECTOR_SIZE; i++) {
    A[i] = i;
    B[i] = VECTOR_SIZE - i;
    C[i] = 0;
  }

  // Create a CL context
  CL cl;
  init_cl(&cl);

  // Create memory buffers on the device for each vector
  cl_mem A_clmem = init_cl_buffer(&cl, CL_MEM_READ_ONLY, A, VECTOR_SIZE);
  cl_mem B_clmem = init_cl_buffer(&cl, CL_MEM_READ_ONLY, B, VECTOR_SIZE);
  cl_mem C_clmem = init_cl_buffer(&cl, CL_MEM_WRITE_ONLY, NULL, VECTOR_SIZE);

  // Create a program and kernel from the source
  cl_program program = CL_CHECK_ERR(clCreateProgramWithSource(cl.context, 1,(const char **)&saxpy_kernel, NULL, &err));
  CL_CHECK(clBuildProgram(program, 1, cl.devices, NULL, NULL, NULL));
  cl_kernel kernel = CL_CHECK_ERR(clCreateKernel(program, "saxpy_kernel", &err));

  // Set the kernel arguments
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(float), &alpha));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &A_clmem));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &B_clmem));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &C_clmem));

  // Execute the kernel
  size_t global_size = VECTOR_SIZE;
  size_t local_size = 64;
  CL_CHECK(clEnqueueNDRangeKernel(cl.command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));

  // Copy result to the host
  CL_CHECK(clEnqueueReadBuffer(cl.command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), C, 0, NULL, NULL));

  // Clean up and wait for all the comands to complete
  CL_CHECK(clFlush(cl.command_queue));
  CL_CHECK(clFinish(cl.command_queue));

  // Display the result
  for (int i = 0; i < VECTOR_SIZE; i++) {
    printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);
  }

  // Release all allocated objects and host buffers
  CL_CHECK(clReleaseKernel(kernel));
  CL_CHECK(clReleaseProgram(program));
  CL_CHECK(clReleaseMemObject(A_clmem));
  CL_CHECK(clReleaseMemObject(B_clmem));
  CL_CHECK(clReleaseMemObject(C_clmem));
  free_cl(&cl);
  free(A);
  free(B);
  free(C);
}
