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

size_t global_size = VECTOR_SIZE;
size_t local_size = 64;


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
  cl_mem wte;
  cl_mem wpe;
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
  cl_kernel add;
  cl_kernel embedding;
  cl_kernel matmul;
} Kernels;

typedef struct {
  cl_platform_id* platforms;
  cl_device_id* devices;
  cl_context context;
  cl_command_queue command_queue;
  Kernels kernels;
} CL;


// CL Initialization

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

void init_cl_kernels(CL* cl, char* cl_source) {
  cl_program program = CL_CHECK_ERR(clCreateProgramWithSource(cl->context, 1, (const char**)&cl_source, NULL, &err));
  CL_CHECK(clBuildProgram(program, 1, cl->devices, NULL, NULL, NULL));
  cl->kernels.add = CL_CHECK_ERR(clCreateKernel(program, "add", &err));
  cl->kernels.embedding = CL_CHECK_ERR(clCreateKernel(program, "embedding", &err));
  cl->kernels.matmul = CL_CHECK_ERR(clCreateKernel(program, "matmul", &err));
  CL_CHECK(clReleaseProgram(program));
}

cl_mem init_cl_buffer(CL* cl, float* buf, size_t size, int mode) {
  cl_mem cl_buf = CL_CHECK_ERR(clCreateBuffer(cl->context, mode, size * sizeof(float), NULL, &err));
  if (buf) CL_CHECK(clEnqueueWriteBuffer(cl->command_queue, cl_buf, CL_TRUE, 0, size * sizeof(float), buf, 0, NULL, NULL));
  return cl_buf;
}


// CL Cleanup

void free_cl(CL* cl) {
  CL_CHECK(clReleaseKernel(cl->kernels.embedding));
  CL_CHECK(clReleaseKernel(cl->kernels.matmul));
  CL_CHECK(clReleaseCommandQueue(cl->command_queue));
  CL_CHECK(clReleaseContext(cl->context));
  free(cl->platforms);
  free(cl->devices);
}


// Model Initialization

cl_mem load_layer(FILE* f, CL* cl, int expected_size) {
  size_t layer_size;
  fread(&layer_size, sizeof(size_t), 1, f);
  assert(layer_size == expected_size);
  float* data = malloc(layer_size * sizeof(float));
  fread(data, sizeof(float), layer_size, f);
  cl_mem cl_data = init_cl_buffer(cl, data, layer_size, CL_MEM_READ_ONLY);
  free(data);
  return cl_data;
}

void init_layernorm(FILE* f, CL* cl, LayerNorm* norm, int num_layers, int size) {
  norm->weight = load_layer(f, cl, num_layers * size);
  norm->bias = load_layer(f, cl, num_layers * size);
  norm->size = size;
}

void init_linear(FILE* f, CL* cl, Linear* layer, int num_layers, int in_size, int out_size) {
  layer->weight = load_layer(f, cl, num_layers * in_size * out_size);
  layer->bias = load_layer(f, cl, num_layers * out_size);
  layer->in_size = in_size;
  layer->out_size = out_size;
}

void init_model(FILE* f, CL* cl, GPT2* model) {
  size_t n_weights, layer_size;
  fread(&n_weights, sizeof(size_t), 1, f);
  fread(&model->config, sizeof(Config), 1, f);

  Config cfg = model->config;
  model->embed_tokens = load_layer(f, cl, cfg.vocab_size * cfg.embed_size);
  model->embed_pos = load_layer(f, cl, cfg.context_size * cfg.embed_size);
  init_layernorm(f, cl, &model->ln1, cfg.num_layers, cfg.embed_size);
  init_layernorm(f, cl, &model->ln2, cfg.num_layers, cfg.embed_size);
  init_linear(f, cl, &model->qkv, cfg.num_layers,  cfg.embed_size, 3 * cfg.embed_size);
  init_linear(f, cl, &model->proj, cfg.num_layers, cfg.embed_size, cfg.embed_size);
  init_linear(f, cl, &model->fc1, cfg.num_layers, cfg.embed_size, 4 * cfg.embed_size);
  init_linear(f, cl, &model->fc2, cfg.num_layers, 4 * cfg.embed_size, cfg.embed_size);
  init_layernorm(f, cl, &model->ln_out, 1, cfg.embed_size);
  model->fc_out = load_layer(f, cl, cfg.embed_size * cfg.vocab_size);
}

void init_state(CL* cl, State* state, Config cfg) {
  state->x = init_cl_buffer(cl, NULL, cfg.embed_size, CL_MEM_READ_WRITE);
  state->wte = init_cl_buffer(cl, NULL, cfg.embed_size, CL_MEM_READ_WRITE);
  state->wpe = init_cl_buffer(cl, NULL, cfg.embed_size, CL_MEM_READ_WRITE);
  state->ln1 = init_cl_buffer(cl, NULL, cfg.embed_size, CL_MEM_READ_WRITE);
  state->qkv = init_cl_buffer(cl, NULL, 3 * cfg.embed_size, CL_MEM_READ_WRITE);
  state->attn = init_cl_buffer(cl, NULL, cfg.context_size, CL_MEM_READ_WRITE);
  state->attn_out = init_cl_buffer(cl, NULL, cfg.embed_size, CL_MEM_READ_WRITE);
  state->proj = init_cl_buffer(cl, NULL, cfg.embed_size, CL_MEM_READ_WRITE);
  state->ln2 = init_cl_buffer(cl, NULL, cfg.embed_size, CL_MEM_READ_WRITE);
  state->fc1 = init_cl_buffer(cl, NULL, 4 * cfg.embed_size, CL_MEM_READ_WRITE);
  state->fc2 = init_cl_buffer(cl, NULL, cfg.embed_size, CL_MEM_READ_WRITE);
  state->out = init_cl_buffer(cl, NULL, cfg.vocab_size, CL_MEM_READ_WRITE);
}


// Helper functions

cl_mem add(CL* cl, cl_mem result, cl_mem x, cl_mem y, size_t n) {
  CL_CHECK(clSetKernelArg(cl->kernels.add, 0, sizeof(cl_mem), &result));
  CL_CHECK(clSetKernelArg(cl->kernels.add, 1, sizeof(cl_mem), &x));
  CL_CHECK(clSetKernelArg(cl->kernels.add, 2, sizeof(cl_mem), &y));
  CL_CHECK(clEnqueueNDRangeKernel(cl->command_queue, cl->kernels.add, 1, NULL, &n, &local_size, 0, NULL, NULL));
  return result;
}

cl_mem embedding(CL* cl, cl_mem result, cl_mem data, size_t idx, size_t embed_size) {
  CL_CHECK(clSetKernelArg(cl->kernels.embedding, 0, sizeof(cl_mem), &result));
  CL_CHECK(clSetKernelArg(cl->kernels.embedding, 1, sizeof(cl_mem), &data));
  CL_CHECK(clSetKernelArg(cl->kernels.embedding, 2, sizeof(int), &idx));
  CL_CHECK(clSetKernelArg(cl->kernels.embedding, 3, sizeof(int), &embed_size));
  CL_CHECK(clEnqueueNDRangeKernel(cl->command_queue, cl->kernels.embedding, 1, NULL, &embed_size, &local_size, 0, NULL, NULL));
  return result;
}


// Forward pass

cl_mem gpt(CL* cl, GPT2* model, State* state, cl_mem past, int past_len, int token) {
  Config cfg = model->config;
  int cache_size = 2 * cfg.context_size * cfg.embed_size;
  cl_mem x, wte, wpe;

  wte = embedding(cl, state->wte, model->embed_tokens, token, cfg.embed_size);
  wpe = embedding(cl, state->wpe, model->embed_pos, past_len, cfg.embed_size);
  x = add(cl, state->x, wte, wpe, cfg.embed_size);
  // for (int i = 0; i < cfg.num_layers; i++) {
  //   x = add(state->x, x, attention(model, state, &past[i * cache_size], past_len, i, norm(state->ln1, &model->ln1, i, x)), cfg.embed_size);
  //   x = add(state->x, x, mlp(model, state, i, norm(state->ln2, &model->ln2, i, x)), cfg.embed_size);
  // }
  // x = matmul(state->out, model->fc_out, norm(state->x, &model->ln_out, 0, x), cfg.embed_size, cfg.vocab_size);
  return x;
}


// Main

int main() {
  CL* cl = malloc(sizeof(CL));
  GPT2* model = malloc(sizeof(GPT2));
  State* state = malloc(sizeof(State));

  // Load the model weights and initialize the model
  FILE* f = fopen("/tmp/weights.bin", "rb");
  if (!f) { fprintf(stderr, "Couldn't open file /tmp/weights.bin\n"); return 1; }
  init_cl(cl);
  init_model(f, cl, model);
  init_state(cl, state, model->config);
  fclose(f);
  Config cfg = model->config;

  // Load the kernel source and initialize the kernels
  f = fopen("gpt2/kernels.cl", "r");
  if (!f) { fprintf(stderr, "Couldn't open file kernels.cl\n"); return 1; }
  fseek(f, 0L, SEEK_END);
  int sz = ftell(f);
  char* cl_source = malloc(sz);
  fseek(f, 0L, SEEK_SET);
  fread(cl_source, 1, sz, f);
  fclose(f);
  init_cl_kernels(cl, cl_source);

  // Forward pass
  cl_mem past = init_cl_buffer(cl, NULL, 2 * cfg.num_layers * cfg.context_size * cfg.embed_size, CL_MEM_READ_WRITE);
  cl_mem result = gpt(cl, model, state, past, 0, 0);

  float* result_host = malloc(cfg.embed_size * sizeof(float));
  CL_CHECK(clEnqueueReadBuffer(cl->command_queue, result, CL_TRUE, 0, cfg.embed_size * sizeof(float), result_host, 0, NULL, NULL));
  CL_CHECK(clFlush(cl->command_queue));
  CL_CHECK(clFinish(cl->command_queue));

  for (int i = 0; i < 3; i++) {
    printf("%f ", result_host[i]);
  }
  printf("\n");
}
