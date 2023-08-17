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
  float* A = (float*)malloc(sizeof(float)*VECTOR_SIZE);
  float* B = (float*)malloc(sizeof(float)*VECTOR_SIZE);
  float* C = (float*)malloc(sizeof(float)*VECTOR_SIZE);
  for (int i = 0; i < VECTOR_SIZE; i++) {
    A[i] = i;
    B[i] = VECTOR_SIZE - i;
    C[i] = 0;
  }

  // Create a CL context
  cl_uint num_platforms;
  CL_CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
  cl_platform_id* platforms = malloc(num_platforms * sizeof(cl_platform_id));
  CL_CHECK(clGetPlatformIDs(num_platforms, platforms, NULL));

  cl_uint num_devices;
  CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices));
  cl_device_id* device_list = malloc(num_devices * sizeof(cl_device_id));
  CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL));

  cl_context context = CL_CHECK_ERR(clCreateContext(NULL, num_devices, device_list, NULL, NULL, &err));
  cl_command_queue command_queue = CL_CHECK_ERR(clCreateCommandQueue(context, device_list[0], 0, &err));

  // Create memory buffers on the device for each vector
  cl_mem A_clmem = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &err));
  cl_mem B_clmem = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &err));
  cl_mem C_clmem = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(float), NULL, &err));

  // Copy A and B to the device
  CL_CHECK(clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL));

  // Create a program and kernel from the source
  cl_program program = CL_CHECK_ERR(clCreateProgramWithSource(context, 1,(const char **)&saxpy_kernel, NULL, &err));
  CL_CHECK(clBuildProgram(program, 1, device_list, NULL, NULL, NULL));
  cl_kernel kernel = CL_CHECK_ERR(clCreateKernel(program, "saxpy_kernel", &err));

  // Set the kernel arguments
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(float), &alpha));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &A_clmem));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &B_clmem));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &C_clmem));

  // Execute the kernel
  size_t global_size = VECTOR_SIZE;
  size_t local_size = 64;
  CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));

  // Copy C to the host
  CL_CHECK(clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), C, 0, NULL, NULL));

  // Clean up and wait for all the comands to complete
  CL_CHECK(clFlush(command_queue));
  CL_CHECK(clFinish(command_queue));

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
  CL_CHECK(clReleaseCommandQueue(command_queue));
  CL_CHECK(clReleaseContext(context));
  free(A);
  free(B);
  free(C);
  free(platforms);
  free(device_list);
}
