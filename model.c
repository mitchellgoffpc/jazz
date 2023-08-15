#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define NUM_LAYERS 12
#define NUM_HEADS 12
#define VOCAB_SIZE 50257
#define EMBED_SIZE 768
#define CONTEXT_SIZE 1024
#define HEAD_SIZE (EMBED_SIZE / NUM_HEADS)
#define CACHE_SIZE (2 * CONTEXT_SIZE * EMBED_SIZE)


// Definitions

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
  float* data = malloc(layer_size*sizeof(float));
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

  model->embed_tokens = load_layer(f, VOCAB_SIZE * EMBED_SIZE);
  model->embed_pos = load_layer(f, CONTEXT_SIZE * EMBED_SIZE);
  init_layernorm(f, &model->ln1, NUM_LAYERS, EMBED_SIZE);
  init_layernorm(f, &model->ln2, NUM_LAYERS, EMBED_SIZE);
  init_linear(f, &model->qkv, NUM_LAYERS,  EMBED_SIZE, 3*EMBED_SIZE);
  init_linear(f, &model->proj, NUM_LAYERS, EMBED_SIZE, EMBED_SIZE);
  init_linear(f, &model->fc1, NUM_LAYERS, EMBED_SIZE, 4*EMBED_SIZE);
  init_linear(f, &model->fc2, NUM_LAYERS, 4*EMBED_SIZE, EMBED_SIZE);
  init_layernorm(f, &model->ln_out, 1, EMBED_SIZE);
  model->fc_out = load_layer(f, EMBED_SIZE * VOCAB_SIZE);
}

void init_state(State* state) {
  state->x = malloc(EMBED_SIZE * sizeof(float));
  state->ln1 = malloc(EMBED_SIZE * sizeof(float));
  state->qkv = malloc(3*EMBED_SIZE * sizeof(float));
  state->attn = malloc(CONTEXT_SIZE * sizeof(float));
  state->attn_out = malloc(EMBED_SIZE * sizeof(float));
  state->proj = malloc(EMBED_SIZE * sizeof(float));
  state->ln2 = malloc(EMBED_SIZE * sizeof(float));
  state->fc1 = malloc(4*EMBED_SIZE * sizeof(float));
  state->fc2 = malloc(EMBED_SIZE * sizeof(float));
  state->out = malloc(EMBED_SIZE * sizeof(float));
}


// Helper Functions

#define ELEMENTWISE(i, n, out, expr) ({ for (int i = 0; i < (n); i++) { out[i] = (expr); } out; })
#define REDUCE(i, n, acc, expr) ({ float acc = 0; for (int i = 0; i < (n); i++) { acc = (expr); } acc; })

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
    out[row] = REDUCE(col, n_cols, acc, acc + A[row*n_cols + col] * X[col]);
  }
  return out;
}
float* matvmul(float* out, float* X, float* A, int n_rows, int n_cols) {
  for (int col = 0; col < n_cols; col++) {
    out[col] = REDUCE(row, n_rows, acc, acc + X[row] * A[row*n_cols + col]);
  }
  return out;
}

float* linear(float* out, Linear* fc, int layer, float* data) {
  out = matmul(out, &fc->weight[layer*fc->in_size*fc->out_size], data, fc->in_size, fc->out_size);
  out = add(out, &fc->bias[layer*fc->out_size], out, fc->out_size);
  return out;
}

float* embedding(float* data, int idx) {
  return &data[idx*EMBED_SIZE];
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
    out[i] = out[i] * norm->weight[layer*norm->size + i] + norm->bias[layer*norm->size + i];
  }
  return out;
}


// Forward pass

float* attention(GPT2* model, State* state, float* past, int past_len, int layer, float* x) {
  float* qkv, *attn, *q, *k, *v;
  qkv = linear(state->qkv, &model->qkv, layer, x);
  q = &qkv[0], k = &qkv[EMBED_SIZE], v = &qkv[2*EMBED_SIZE];

  for (int i = 0; i < NUM_HEADS; i++) {
    float* k_past = &past[i*CONTEXT_SIZE*HEAD_SIZE];
    float* v_past = &past[NUM_HEADS*CONTEXT_SIZE*HEAD_SIZE + i*CONTEXT_SIZE*HEAD_SIZE];
    memcpy(&k_past[past_len*HEAD_SIZE], &k[i*HEAD_SIZE], HEAD_SIZE*sizeof(float));
    memcpy(&v_past[past_len*HEAD_SIZE], &v[i*HEAD_SIZE], HEAD_SIZE*sizeof(float));

    attn = matmul(state->attn, k_past, &q[i*HEAD_SIZE], HEAD_SIZE, past_len+1);
    attn = scale(state->attn, (1.0 / sqrt(HEAD_SIZE)), past_len+1);
    attn = softmax(state->attn, past_len+1);
    attn = matvmul(&state->attn_out[i*HEAD_SIZE], attn, v_past, past_len+1, HEAD_SIZE);
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
  float* x;
  x = add(state->x, embedding(model->embed_tokens, token), embedding(model->embed_pos, past_len), EMBED_SIZE);
  for (int i = 0; i < NUM_LAYERS; i++) {
    x = add(state->x, x, attention(model, state, &past[i*CACHE_SIZE], past_len, i, norm(state->ln1, &model->ln1, i, x)), EMBED_SIZE);
    x = add(state->x, x, mlp(model, state, i, norm(state->ln2, &model->ln2, i, x)), EMBED_SIZE);
  }
  x = matmul(state->out, model->fc_out, norm(state->x, &model->ln_out, 0, x), EMBED_SIZE, VOCAB_SIZE);
  return x;
}


// Main

int main() {
  GPT2* model = malloc(sizeof(GPT2));
  State* state = malloc(sizeof(State));
  float* past = malloc(2 * NUM_LAYERS * CONTEXT_SIZE * EMBED_SIZE * sizeof(float));

  FILE* f = fopen("/tmp/weights.bin", "rb");
  assert(f);
  init_model(f, model);
  init_state(state);

  // The capital of Germany is Berlin. The capital of France is ...
  int tokens[12] = {464, 3139, 286, 4486, 318, 11307, 13, 383, 3139, 286, 4881, 318};
  for (int i = 0; i < 12; i++) {
    gpt(model, state, past, i, tokens[i]);
  }

  int argmax = 0;
  for (int i = 0; i < VOCAB_SIZE; i++) {
    argmax = state->out[i] > state->out[argmax] ? i : argmax;
  }
  assert(argmax == 6342);  // Paris
  printf("DONE\n");
}
