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
  int n = a->len_size;
  int len_logical = 1;
  for(int i = 0; i < n; i++) {
    len_logical *= *(a->size + i);
  }
  int *suf = malloc(n * sizeof(int));
  *(suf + n - 1) = 1;
  for(int i = n - 2; i >= 0; i--) {
    *(suf + i) = *(suf + i + 1) * *(a->size + i + 1);
  }
  int *idx = malloc(n * sizeof(int));
  for(int i = 0; i < len_logical; i++) {
    for(int j = 0; j < n; j++) {
      *(idx + j) = (i / *(suf + j)) % *(a->size + j);
    }
    for(int j = 0; j < n; j++) {
      if(i % (*(a->size + j) * *(suf + j)) == 0) {
        printf("(");
      }
    }
    printf("%.1f, ", *atto_index(a, idx));
    int first = 1;
    for(int j = 0; j < n; j++) {
      if((i + 1) % (*(a->size + j) * *(suf + j)) == 0) {
        if(first) {
          printf("\b\b)");
          first = 0;
        } else {
          printf(")");
        }
        if(j == n - 1) {
          if(i < len_logical - 1) {
            printf(",\n");
          } else {
            printf("\n");
          }
        }
      }
    }
  }
  free(suf);
  free(idx);
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
  b->size = malloc(lensz * sizeof(int));
  memcpy(b->size, size, lensz * sizeof(int));
  b->len_size = lensz;
  b->strides = atto_calculate_contiguous_strides(size, lensz);
  return b;
}

float *atto_index(Tensor *a, int *idx) {
  for(int i = 0; i < a->len_size; i++) {
    if(*(idx + i) < 0 || *(idx + i) >= *(a->size + i)) {
      fprintf(stderr, "IndexError: %d is out of bounds in dimension %d\n", *(idx + i), i);
      return nullptr;
    }
  }
  int flat_idx = 0;
  for(int i = 0; i < a->len_size; i++) {
    flat_idx += *(idx + i) * *(a->strides + i);
  }
  return (a->data + flat_idx);
}

Tensor *atto_broadcast(Tensor *a, int *size, int lensz) {
  Tensor *b = malloc(sizeof(Tensor));
  b->data = a->data;
  b->len_data = a->len_data;
  b->size = malloc(lensz * sizeof(int));
  memcpy(b->size, size, lensz * sizeof(int));
  b->len_size = lensz;
  b->strides = calloc(lensz, sizeof(int));
  for(int i = 0; i < a->len_size; i++) {
    *(b->strides + lensz - 1 - i) = *(b->size + lensz - 1 - i) > *(a->size + a->len_size - 1 - i) ? 0 : *(a->strides + a->len_size - 1 - i);
  }
  return b;
}
