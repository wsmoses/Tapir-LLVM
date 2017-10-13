#include <cilk/cilk.h>
#include <stdlib.h>
#include "../common.h"

const int NUM_REPEATS = 200;
const int SIZE = 100000000;

#define ITER cilk_for (int i = 0; i < SIZE; i++)

int main() {
  int *A = (int*) malloc(SIZE * sizeof(int));
  int *B = (int*) malloc(SIZE * sizeof(int));
  int *C = (int*) malloc(SIZE * sizeof(int));

  profile_start();

  for (int k = 0; k < NUM_REPEATS; k++) {
#ifdef OPT
    ITER {
      A[i] += B[i];
      A[i] += C[i];
    }
#else
    ITER {
      A[i] += B[i];
    }
    ITER {
      A[i] += C[i];
    }
#endif
  }

  profile_end();

  return 0;
}
