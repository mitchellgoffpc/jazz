#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


// Definitions

typedef struct {
  size_t num_layers;
  size_t num_heads;
  size_t embed_size;
  size_t vocab_size;
  size_t context_size;
} Config;

typedef struct {
  float* weight;
  float* bias;
  int size;
} LayerNorm;

typedef struct {
  float* weight;
  float* bias;
  int in_size;
  int out_size;
} Linear;

typedef struct {
  Config config;
  float* embed_tokens;
  float* embed_pos;
  LayerNorm ln1;
  LayerNorm ln2;
  Linear qkv;
  Linear proj;
  Linear fc1;
  Linear fc2;
  LayerNorm ln_out;
  float* fc_out;
} GPT2;

typedef struct {
  float* past;
  float* x;
  float* ln1;
  float* qkv;
  float* attn;
  float* attn_out;
  float* proj;
  float* ln2;
  float* fc1;
  float* fc2;
  float* out;
} State;


// Initialization

float* load_layer(FILE* f, int expected_size) {
  size_t layer_size;
  fread(&layer_size, sizeof(size_t), 1, f);
  assert(layer_size == expected_size);
  float* data = malloc(layer_size * sizeof(float));
  fread(data, sizeof(float), layer_size, f);
  return data;
}

void init_layernorm(FILE* f, LayerNorm* norm, int num_layers, int size) {
  norm->weight = load_layer(f, num_layers * size);
  norm->bias = load_layer(f, num_layers * size);
  norm->size = size;
}

void init_linear(FILE* f, Linear* layer, int num_layers, int in_size, int out_size) {
  layer->weight = load_layer(f, num_layers * in_size * out_size);
  layer->bias = load_layer(f, num_layers * out_size);
  layer->in_size = in_size;
  layer->out_size = out_size;
}

void init_model(FILE* f, GPT2* model) {
  size_t n_weights, layer_size;
  fread(&n_weights, sizeof(size_t), 1, f);
  fread(&model->config, sizeof(Config), 1, f);

  Config cfg = model->config;
  model->embed_tokens = load_layer(f, cfg.vocab_size * cfg.embed_size);
  model->embed_pos = load_layer(f, cfg.context_size * cfg.embed_size);
  init_layernorm(f, &model->ln1, cfg.num_layers, cfg.embed_size);
  init_layernorm(f, &model->ln2, cfg.num_layers, cfg.embed_size);
  init_linear(f, &model->qkv, cfg.num_layers,  cfg.embed_size, 3 * cfg.embed_size);
  init_linear(f, &model->proj, cfg.num_layers, cfg.embed_size, cfg.embed_size);
  init_linear(f, &model->fc1, cfg.num_layers, cfg.embed_size, 4 * cfg.embed_size);
  init_linear(f, &model->fc2, cfg.num_layers, 4 * cfg.embed_size, cfg.embed_size);
  init_layernorm(f, &model->ln_out, 1, cfg.embed_size);
  model->fc_out = load_layer(f, cfg.embed_size * cfg.vocab_size);
}

void init_state(State* state, Config cfg) {
  state->x = malloc(cfg.embed_size * sizeof(float));
  state->ln1 = malloc(cfg.embed_size * sizeof(float));
  state->qkv = malloc(3 * cfg.embed_size * sizeof(float));
  state->attn = malloc(cfg.context_size * sizeof(float));
  state->attn_out = malloc(cfg.embed_size * sizeof(float));
  state->proj = malloc(cfg.embed_size * sizeof(float));
  state->ln2 = malloc(cfg.embed_size * sizeof(float));
  state->fc1 = malloc(4 * cfg.embed_size * sizeof(float));
  state->fc2 = malloc(cfg.embed_size * sizeof(float));
  state->out = malloc(cfg.vocab_size * sizeof(float));
}


// Helper Functions

#define ELEMENTWISE(i, n, out, expr) ({ \
  for (int i = 0; i < (n); i++) { out[i] = (expr); } \
  out; \
})
#define REDUCE(i, n, acc, expr) ({ \
  float acc = 0; \
  for (int i = 0; i < (n); i++) { acc = (expr); } \
  acc; \
})

float* add(float* out, float* x, float* y, int n) {
  return ELEMENTWISE(i, n, out, x[i] + y[i]);
}
float* shift(float* x, float c, int n) {
  return ELEMENTWISE(i, n, x, c + x[i]);
}
float* scale(float* x, float c, int n) {
  return ELEMENTWISE(i, n, x, c * x[i]);
}
float* vexp(float* x, int n) {
  return ELEMENTWISE(i, n, x, (float) exp(x[i]));
}
float* gelu(float* x, int n) {
  float sqrt2overPI = sqrt(2 / M_PI);
  return ELEMENTWISE(i, n, x, 0.5 * x[i] * (1 + tanh(sqrt2overPI * (x[i] + 0.044715 * x[i]*x[i]*x[i]))));
}

float sum(float* x, int n) {
  return REDUCE(i, n, acc, acc + x[i]);
}
float max(float* x, int n) {
  return REDUCE(i, n, acc, x[i] > acc ? x[i] : acc);
}

float* matmul(float* out, float* A, float* X, int n_cols, int n_rows) {
  for (int row = 0; row < n_rows; row++) {
    out[row] = REDUCE(col, n_cols, acc, acc + A[row * n_cols + col] * X[col]);
  }
  return out;
}
float* matvmul(float* out, float* X, float* A, int n_rows, int n_cols) {
  memset(out, 0, n_cols*sizeof(float));
  for (int row = 0; row < n_rows; row++) {
    for (int col = 0; col < n_cols; col++) {
      out[col] += X[row] * A[row * n_cols + col];
    }
  }
  return out;
}

float* linear(float* out, Linear* fc, int layer, float* data) {
  out = matmul(out, &fc->weight[layer * fc->in_size * fc->out_size], data, fc->in_size, fc->out_size);
  out = add(out, &fc->bias[layer * fc->out_size], out, fc->out_size);
  return out;
}

float* embedding(float* data, int idx, int embed_size) {
  return &data[idx * embed_size];
}

float* softmax(float* x, int n) {
  x = shift(x, -max(x, n), n);
  x = vexp(x, n);
  x = scale(x, 1.0 / sum(x, n), n);
  return x;
}

float* norm(float* out, LayerNorm* norm, int layer, float* x) {
  float mean = 0, var = 0;
  for (int i = 0; i < norm->size; i++) {
    mean += x[i] / norm->size;
  }
  for (int i = 0; i < norm->size; i++) {
    var += (x[i] - mean) * (x[i] - mean) / norm->size;
  }
  float std = sqrt(var) + 1e-5; // epsilon
  for (int i = 0; i < norm->size; i++) {
    out[i] = (x[i] - mean) / std;
    out[i] = out[i] * norm->weight[layer * norm->size + i] + norm->bias[layer * norm->size + i];
  }
  return out;
}


// Forward pass

float* attention(GPT2* model, State* state, float* past, int past_len, int layer, float* x) {
  Config cfg = model->config;
  int cache_size = 2 * cfg.context_size * cfg.embed_size;  // size per layer
  int head_size = cfg.embed_size / cfg.num_heads;
  float* qkv, *attn, *q, *k, *v;

  qkv = linear(state->qkv, &model->qkv, layer, x);
  q = &qkv[0 * cfg.embed_size];
  k = &qkv[1 * cfg.embed_size];
  v = &qkv[2 * cfg.embed_size];

  for (int i = 0; i < cfg.num_heads; i++) {
    float* k_past = &past[(layer * cache_size) + (0 * cfg.num_heads * cfg.context_size * head_size) + (i * cfg.context_size * head_size)];
    float* v_past = &past[(layer * cache_size) + (1 * cfg.num_heads * cfg.context_size * head_size) + (i * cfg.context_size * head_size)];
    memcpy(&k_past[past_len * head_size], &k[i * head_size], head_size * sizeof(float));
    memcpy(&v_past[past_len * head_size], &v[i * head_size], head_size * sizeof(float));

    attn = matmul(state->attn, k_past, &q[i * head_size], head_size, past_len + 1);
    attn = scale(state->attn, (1.0 / sqrt(head_size)), past_len + 1);
    attn = softmax(state->attn, past_len + 1);
    attn = matvmul(&state->attn_out[i * head_size], attn, v_past, past_len + 1, head_size);
  }

  x = linear(state->proj, &model->proj, layer, state->attn_out);
  return x;
}

float* mlp(GPT2* model, State* state, int layer, float* x) {
  x = linear(state->fc1, &model->fc1, layer, x);
  x = linear(state->fc2, &model->fc2, layer, gelu(x, model->fc1.out_size));
  return x;
}

float* gpt(GPT2* model, State* state, float* past, int past_len, int token) {
  Config cfg = model->config;
  float* x;

  x = add(state->x, embedding(model->embed_tokens, token, cfg.embed_size), embedding(model->embed_pos, past_len, cfg.embed_size), cfg.embed_size);
  for (int i = 0; i < cfg.num_layers; i++) {
    x = add(state->x, x, attention(model, state, past, past_len, i, norm(state->ln1, &model->ln1, i, x)), cfg.embed_size);
    x = add(state->x, x, mlp(model, state, i, norm(state->ln2, &model->ln2, i, x)), cfg.embed_size);
  }
  x = matmul(state->out, model->fc_out, norm(state->x, &model->ln_out, 0, x), cfg.embed_size, cfg.vocab_size);
  return x;
}


// Main

#define NUM_SAMPLES 50
#define BENCHMARK(n, expr) ({ \
  clock_t start_time = clock(); \
  for (int i = 0; i < (n); i++) { expr; } \
  double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC; \
  printf("DONE, %f.2 it/s\n", n / elapsed_time); \
})

int main() {
  GPT2* model = malloc(sizeof(GPT2));
  State* state = malloc(sizeof(State));

  FILE* f = fopen("/tmp/weights.bin", "rb");
  if (!f) { fprintf(stderr, "Couldn't open file /tmp/weights.bin\n"); return 1; }
  init_model(f, model);
  init_state(state, model->config);
  fclose(f);
  Config cfg = model->config;

  // The capital of Germany is Berlin. The capital of France is ...
  printf("TESTING...\n");
  float* past = malloc(2 * cfg.num_layers * cfg.context_size * cfg.embed_size * sizeof(float));
  int tokens[12] = {464, 3139, 286, 4486, 318, 11307, 13, 383, 3139, 286, 4881, 318};
  for (int i = 0; i < 12; i++) {
    gpt(model, state, past, i, tokens[i]);
  }

  int argmax = 0;
  for (int i = 0; i < cfg.vocab_size; i++) {
    argmax = state->out[i] > state->out[argmax] ? i : argmax;
  }
  assert(argmax == 6342);  // Paris
  printf("DONE\n");

  printf("BENCHMARKING, T=1...\n");
  BENCHMARK(NUM_SAMPLES, gpt(model, state, past, 0, 0));

  printf("BENCHMARKING, T=%lu...\n", cfg.context_size);
  BENCHMARK(NUM_SAMPLES, gpt(model, state, past, cfg.context_size-1, 0));
}
