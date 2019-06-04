#include "ann.h"

/* Creates and returns a new ann. */
ann_t *ann_create(int num_layers, int *layer_outputs)
{
  ann_t *ann = (ann_t*)malloc(sizeof(ann_t*));
  if (num_layers < 2) {
    return NULL;
  }
  ann->input_layer = layer_create();
  if (!layer_init(ann->input_layer, layer_outputs[0], NULL)) {
    return NULL;
  }
  layer_t *prev = ann->input_layer;
  for (int i = 1; i < num_layers; i++) {
    layer_t *current_layer = layer_create();
    if (!layer_init(current_layer, layer_outputs[i], prev)) {
      return NULL;
    }
    prev->next = current_layer;
    prev = current_layer;
  }
  ann->output_layer = prev;
  return ann;
}

/* Frees the space allocated to ann. */
void ann_free(ann_t *ann)
{
  layer_t *pointer = ann->input_layer;
  while (pointer) {
    layer_free(pointer);
    layer_t *next = pointer->next;
    free(pointer);
    pointer = next;
  }
  free(ann);
}

/* Forward run of given ann with inputs. */
void ann_predict(ann_t const *ann, double const *inputs)
{
  layer_t *input_layer = ann->input_layer;
  input_layer->outputs = inputs;
  layer_t *current = input_layer->next;
  while (current) {
    layer_compute_outputs(current);
    current = current->next;
  }
}

/* Trains the ann with single backprop update. */
void ann_train(ann_t const *ann, double const *inputs, double const *targets, double l_rate)
{
  /* Sanity checks. */
  assert(ann != NULL);
  assert(inputs != NULL);
  assert(targets != NULL);
  assert(l_rate > 0);

  /* Run forward pass. */
  ann_predict(ann, inputs);

  /**** PART 2 - QUESTION 4 ****/

  /* 3 MARKS */
}