#include <cilk/cilk.h>
#include "../common.h"

const int NUM_REPEATS = 10000;
const int SIZE_I = 1000;
const int SIZE_J = 1000;
const int SIZE = SIZE_I * SIZE_J;

int main() {
  unsigned int *A = (unsigned int*) malloc(SIZE * sizeof(unsigned int));
  unsigned int *B = (unsigned int*) malloc(SIZE * sizeof(unsigned int));

  profile_start();

  for (int k = 0; k < NUM_REPEATS; k++) {
#ifdef OPT
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

  profile_end();

  return A[A[0] % SIZE];
}




