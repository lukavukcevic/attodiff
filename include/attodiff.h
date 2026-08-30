#ifndef ATTODIFF_H
#define ATTODIFF_H 1
typedef struct {
  float *data;
  int len_data;
  int *size;
  int len_size;
  int *strides;
} Tensor;
int *atto_calculate_contiguous_strides(int *size, int lensz);
Tensor *atto_ones(int *size, int lensz);
void atto_print_flat(char t, void *arr, int len);
void atto_print(Tensor *a);
int atto_are_broadcastable(Tensor *a, Tensor *b);
Tensor *atto_view(Tensor *a, int *size, int lensz);
float *atto_index(Tensor *a, int *idx);
#endif
