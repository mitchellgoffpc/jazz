#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_LAYERS 12
#define NUM_HEADS 12
#define VOCAB_SIZE 50257
#define EMBED_SIZE 768
#define CONTEXT_SIZE 1024


// Definitions

typedef struct {
  float* weight;
  float* bias;
  int in_size;
  int out_size;
} Linear;

typedef struct {
  float* embed_tokens;
  float* embed_pos;
  Linear qkv;
  Linear proj;
  Linear fc1;
  Linear fc2;
  float* fc_out;
} GPT2;

typedef struct {
  float* past;
  float* x;
  float* ln1;
  float* qkv;
  float* attn;
  float* proj;
  float* ln2;
  float* fc1;
  float* fc2;
  float* out;
} State;


// Initialization

void init_linear(Linear* layer, int in_size, int out_size) {
  layer->weight = malloc(in_size * out_size * sizeof(float));
  layer->bias = malloc(out_size * sizeof(float));
  layer->in_size = in_size;
  layer->out_size = out_size;
}

void init_model(GPT2* model) {
  model->embed_tokens = malloc(VOCAB_SIZE * EMBED_SIZE * sizeof(float));
  model->embed_pos = malloc(CONTEXT_SIZE * EMBED_SIZE * sizeof(float));
  init_linear(&model->qkv, EMBED_SIZE, 3*EMBED_SIZE);
  init_linear(&model->proj, EMBED_SIZE, EMBED_SIZE);
  init_linear(&model->fc1, EMBED_SIZE, 4*EMBED_SIZE);
  init_linear(&model->fc2, 4*EMBED_SIZE, EMBED_SIZE);
  model->fc_out = malloc(EMBED_SIZE * VOCAB_SIZE * sizeof(float));
}

void init_state(State* state) {
  state->x = malloc(EMBED_SIZE * sizeof(float));
  state->ln1 = malloc(EMBED_SIZE * sizeof(float));
  state->qkv = malloc(3*EMBED_SIZE * sizeof(float));
  state->attn = malloc(CONTEXT_SIZE * sizeof(float));
  state->proj = malloc(EMBED_SIZE * sizeof(float));
  state->ln2 = malloc(EMBED_SIZE * sizeof(float));
  state->fc1 = malloc(EMBED_SIZE * sizeof(float));
  state->fc2 = malloc(EMBED_SIZE * sizeof(float));
  state->out = malloc(EMBED_SIZE * sizeof(float));
}


// Helper Functions

#define ELEMENTWISE(i, n, out, expr) ({ for (int (i) = 0; (i) < (n); i++) { out[i] = (expr); } out; })
#define REDUCE(i, n, acc, expr) ({ float acc = 0; for (int (i) = 0; (i) < (n); i++) { acc = (expr); } acc; })

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

float* linear(float* out, Linear* layer, float* data) {
  out = matmul(out, layer->weight, data, layer->in_size, layer->out_size);
  out = add(out, layer->bias, out, layer->out_size);
  return out;
}

float* embedding(float* data, int idx) {
  return &data[idx*EMBED_SIZE];
}

float* softmax(float* x, int n) {
  x = shift(x, -max(x, n), n);
  x = vexp(x, n);
  x = scale(x, 1 / sum(x, n), n);
  return x;
}

float* norm(float* x) {
  float mean = 0, std = 0;
  for (int i = 0; i < EMBED_SIZE; i++) {
    mean += x[i] / EMBED_SIZE;
  }
  for (int i = 0; i < EMBED_SIZE; i++) {
    std += (x[i] - mean) * (x[i] - mean);
  }
  for (int i = 0; i < EMBED_SIZE; i++) {
    x[i] = (x[i] - mean) / std;
  }
  return x;
}


// Forward pass

float* attention(GPT2* model, State* state, float* past, int past_len, int layer, float* x) {
  float* qkv, *attn, *q, *k, *v;
  qkv = linear(state->qkv, &model->qkv, x);
  q = &qkv[0], k = &qkv[EMBED_SIZE], v = &qkv[2*EMBED_SIZE];
  attn = matmul(state->attn, k,  q, EMBED_SIZE, past_len);
  attn = scale(attn, (1.0 / sqrt(EMBED_SIZE)), past_len);
  attn = softmax(attn, past_len);
  x = matmul(state->attn, v, attn, past_len, EMBED_SIZE);
  x = linear(state->proj, &model->proj, x);
  return x;
}

float* mlp(GPT2* model, State* state, int layer, float* x) {
  x = linear(state->fc1, &model->fc1, x);
  x = linear(state->fc2, &model->fc2, gelu(x, EMBED_SIZE));
  return x;
}

float* gpt(GPT2* model, State* state, float* past, int past_len, int token) {
  float* x;
  x = add(state->x, embedding(model->embed_tokens, token), embedding(model->embed_pos, past_len), EMBED_SIZE);
  for (int i = 0; i < NUM_LAYERS; i++) {
    x = add(state->x, x, attention(model, state, past, past_len, i, norm(x)), EMBED_SIZE);
    x = add(state->x, x, mlp(model, state, i, norm(x)), EMBED_SIZE);
  }
  return x;
}


// Main

int main() {
  // deserialize()
  GPT2* model = malloc(sizeof(GPT2));
  State* state = malloc(sizeof(State));
  float* past = malloc(2 * NUM_LAYERS * CONTEXT_SIZE * EMBED_SIZE * sizeof(float));

  init_model(model);
  init_state(state);

  int context[512] = {0};
  gpt(model, state, past, 0, 0);
  printf("DONE\n");
}
