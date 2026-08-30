#ifndef ATTODIFF_H
#define ATTODIFF_H 1
typedef struct {
  float *data;
  int len_data;
  int *size;
  int len_size;
  int *strides;
} Tensor;
Tensor *atto_ones(int *size, int lensz);
void atto_print_flat(char t, void *arr, int len);
void atto_print(Tensor *a);
int atto_are_broadcastable(Tensor *a, Tensor *b);
#endif
