#include <cilk/cilk.h>
#include <stdlib.h>

const int NUM_REPEATS = 10000;
const int SIZE_I = 1000;
const int SIZE_J = 1000;
const int SIZE = SIZE_I * SIZE_J;

#define FLAT

int main() {
  int *A = (int*) malloc(SIZE * sizeof(int));
  int *B = (int*) malloc(SIZE * sizeof(int));

  for (int k = 0; k < NUM_REPEATS; k++) {
#ifdef FLAT
    cilk_for (int i = 0; i < SIZE; i++) {
    A[i] += B[i];
  }
#else
    cilk_for (int i = 0; i < SIZE_I; i++) {
      cilk_for (int j = 0; j < SIZE_J; j++) {
        A[SIZE_I * i + j] += B[SIZE_I * i + j];
      }
    }
#endif
  }
  return 0;
}




