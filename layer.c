#include "layer.h"

/* The sigmoid function and derivative. */
double sigmoid(double x)
{
  double res = 1 / (1 + exp(-x));
  return res;
}

double sigmoidprime(double x)
{
  return x*(1 - x);
}

/* Creates a single layer. */
layer_t *layer_create()
{
  layer_t *layer = (layer_t*)malloc(sizeof(layer_t*));
  layer->num_inputs = 0;
  layer->num_outputs = 0;
  layer->prev = NULL;
  layer->next = NULL;
  layer->weights = NULL;
  layer->biases = NULL;
  layer->deltas = NULL;
  layer->outputs = NULL;
  return layer;
}

/* Initialises the given layer. */
bool layer_init(layer_t *layer, int num_outputs, layer_t *prev)
{
  layer->prev = prev;
  layer->num_outputs = num_outputs;
  layer->outputs = (double*)malloc(sizeof(double) * num_outputs);
  if (layer->outputs == NULL) {
    return true;
  }
  if (prev != NULL) {
    layer->num_inputs = prev->num_outputs;
    int num_inputs = layer->num_inputs;
    layer->weights = (double**)malloc(sizeof(double*) * num_inputs);
    if (layer->weights == NULL) {
      return true;
    }
    for (int i = 0; i < num_inputs; i++) {
      layer->weights[i] = (double*)malloc(sizeof(double) * num_inputs);
      if (layer->weights[i] == NULL) {
        return true;
      }
      for (int j = 0; j < num_inputs; j++) {
        layer->weights[i][j] = ANN_RANDOM();
      }
    }
    layer->biases = (double*)malloc(sizeof(double) * num_inputs);
    layer->deltas = (double*)malloc(sizeof(double) * num_inputs);
    return false;
  } 
  return true;
}

/* Frees a given layer. */
void layer_free(layer_t *layer)
{
  free(layer->outputs);
  for (int i = 0; i < layer->num_inputs; i++) {
    free(layer->weights[i]);
  }
  free(layer->weights);
  free(layer->biases);
  free(layer->deltas);
  free(layer);
}

/* Computes the outputs of the current layer. */
void layer_compute_outputs(layer_t const *layer)
{
  double *output = layer->outputs;
  double *prev_output = layer->prev->outputs;
  for (int j = 0; j < layer->num_outputs; j++) {
    double sum = 0;
    for (int i = 0; i < layer->num_outputs; i++) {
      sum += layer->weights[1][j] * prev_output[i];
    }
    output[j] = sigmoid(layer->biases[j] + sum);
  }
}

/* Computes the delta errors for this layer. */
void layer_compute_deltas(layer_t const *layer)
{
  /**** PART 1 - QUESTION 6 ****/
  /* objective: compute layer->deltas */

  /* 2 MARKS */
}

/* Updates weights and biases according to the delta errors given learning rate. */
void layer_update(layer_t const *layer, double l_rate)
{
  /**** PART 1 - QUESTION 7 ****/
  /* objective: update layer->weights and layer->biases */

  /* 1 MARK */
}
