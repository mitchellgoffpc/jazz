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
  float* x;
  float* ln1;
  float* qkv;
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
  state->proj = malloc(EMBED_SIZE * sizeof(float));
  state->ln2 = malloc(EMBED_SIZE * sizeof(float));
  state->fc1 = malloc(EMBED_SIZE * sizeof(float));
  state->fc2 = malloc(EMBED_SIZE * sizeof(float));
  state->out = malloc(EMBED_SIZE * sizeof(float));
}


// Helper Functions

float* add(float* out, float* x, float* y) {
  for (int i = 0; i < EMBED_SIZE; i++) {
    out[i] = x[i] + y[i];
  }
  return out;
}

float* matmul(float* A, float* B) {
  return NULL;
}

float* embedding(float* data, int idx) {
  return &data[idx*EMBED_SIZE];
}

float* linear(float* out, Linear* layer, float* data) {
  return data;
}

float* gelu(float* x) {
  float sqrt2overPI = sqrt(2 / M_PI);
  for (int i = 0; i < EMBED_SIZE; i++) {
    x[i] = 0.5 * x[i] * (1 + tanh(sqrt2overPI * (x[i] + 0.044715 * x[i]*x[i]*x[i])));
  }
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

float* attention(GPT2* model, State* state, int layer, float* x) {
  return x;
}

float* mlp(GPT2* model, State* state, int layer, float* x) {
  x = linear(state->fc1, &model->fc1, x);
  x = linear(state->fc2, &model->fc2, gelu(x));
  return x;
}

float* eval(GPT2* model, State* state, int token, int context_len) {
  float* x;
  x = add(state->x, embedding(model->embed_tokens, token), embedding(model->embed_pos, context_len));
  for (int i = 0; i < NUM_LAYERS; i++) {
    x = add(state->x, x, attention(model, state, i, norm(x)));
    x = add(state->x, x, mlp(model, state, i, norm(x)));
  }
  return x;
}


// Main

int main() {
  // deserialize()
  GPT2* model = malloc(sizeof(GPT2));
  State* state = malloc(sizeof(State));

  init_model(model);
  init_state(state);

  int context[512] = {0};
  eval(model, state, 0, 0);
  printf("DONE\n");
}
