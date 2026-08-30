#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/attodiff.h"

int *atto_calculate_contiguous_strides(int *size, int lensz) {
  int *strides = malloc(lensz * sizeof(int));
  for(int i = lensz - 1; i >= 0; i--) {
    *(strides + i) = 1;
    for(int j = i + 1; j < lensz; j++) {
      *(strides + i) *= *(size + j);
    }
  }
  return strides;
}

Tensor *atto_ones(int *size, int lensz) {
  Tensor *a = malloc(sizeof(Tensor));
  a->len_data = 1;
  for(int i = 0; i < lensz; i++) {
    a->len_data *= *(size + i);
  }
  a->data = malloc(a->len_data * sizeof(float));
  for(int i = 0; i < a->len_data; i++) {
    *(a->data + i) = 1;
  }
  a->size = malloc(lensz * sizeof(int));
  memcpy(a->size, size, lensz * sizeof(int));
  a->len_size = lensz; 
  a->strides = atto_calculate_contiguous_strides(size, lensz);
  return a;
}

void atto_print_flat(char t, void *arr, int len) {
  switch(t) {
    case 'd':
      printf("(");
      for(int i = 0; i < len; i++) {
        printf("%d, ", ((int*)arr)[i]);
      }
      printf("\b\b)\n");
      break;
    case 'f':
      printf("(");
      for(int i = 0; i < len; i++) {
        printf("%.1f, ", ((float*)arr)[i]);
      }
      printf("\b\b)\n");
      break;
  }
}

void atto_print(Tensor *a) {
  for(int i = 0; i < a->len_data; i++) {
    for(int j = 0; j < a->len_size; j++) {
      if(i % (*(a->size + j) * *(a->strides + j)) == 0) {
        printf("(");
      }
    }
    printf("%.1f, ", *(a->data + i));
    int first = 1;
    for(int j = 0; j < a->len_size; j++) {
      if((i + 1) % (*(a->size + j) * *(a->strides + j)) == 0) {
        if(first) {
          printf("\b\b)");
          first = 0;
        } else {
          printf(")");
        }
        if(j == a->len_size - 1) {
          if(i < a->len_data - 1) {
            printf(",\n");
          } else {
            printf("\n");
          }
        }
      }
    }
  }
}

int atto_are_broadcastable(Tensor *a, Tensor *b) {
  int als = a->len_size - 1;
  int bls = b->len_size - 1;
  int len_min = als < bls ? als : bls;
  for(int i = 0; i < len_min; i++) {
    int asz = a->size[als - i];
    int bsz = b->size[bls - i];
    if((asz != bsz) && (asz != 1) && (bsz != 1)) {
      return 0;
    }
  }
  if(len_min == 0) {
    int asz = a->size[als];
    int bsz = b->size[bls];
    if((asz != bsz) && (asz != 1) && (bsz != 1)) {
      return 0;
    } else {
      return 1;
    }
  }
  return 1;
}
;
Tensor *atto_view(Tensor *a, int *size, int lensz) {
  Tensor *b = malloc(sizeof(Tensor));
  b->data = a->data;
  b->len_data = a->len_data;
  b->size = size;
  b->len_size = lensz;
  b->strides = atto_calculate_contiguous_strides(size, lensz);
  return b;
}
