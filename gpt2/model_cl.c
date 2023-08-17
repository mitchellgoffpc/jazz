#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define VECTOR_SIZE 1024

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
  cl_int clStatus;

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

  // Get platform and device information
  cl_uint num_platforms;
  clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
  cl_platform_id* platforms = malloc(num_platforms * sizeof(cl_platform_id));
  clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

  // Get the devices list and choose the device you want to run on
  cl_uint num_devices;
  clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  cl_device_id* device_list = malloc(num_devices * sizeof(cl_device_id));
  clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

  // Create a CL context for the device
  cl_context context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

  // Create memory buffers on the device for each vector
  cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
  cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
  cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);

  // Copy A and B to the device
  clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL);

  // Create a program and kernel from the source
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy_kernel, NULL, &clStatus);
  clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);

  // Set the kernel arguments
  clStatus = clSetKernelArg(kernel, 0, sizeof(float), &alpha);
  clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), &A_clmem);
  clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), &B_clmem);
  clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), &C_clmem);

  // Execute the kernel
  size_t global_size = VECTOR_SIZE;
  size_t local_size = 64;
  clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

  // Copy C to the host
  clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), C, 0, NULL, NULL);

  // Clean up and wait for all the comands to complete
  clStatus = clFlush(command_queue);
  clStatus = clFinish(command_queue);

  // Display the result
  for (int i = 0; i < VECTOR_SIZE; i++) {
    printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);
  }

  // Release all allocated objects and host buffers
  clStatus = clReleaseKernel(kernel);
  clStatus = clReleaseProgram(program);
  clStatus = clReleaseMemObject(A_clmem);
  clStatus = clReleaseMemObject(B_clmem);
  clStatus = clReleaseMemObject(C_clmem);
  clStatus = clReleaseCommandQueue(command_queue);
  clStatus = clReleaseContext(context);
  free(A);
  free(B);
  free(C);
  free(platforms);
  free(device_list);
}
