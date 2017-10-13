#include <cilk/cilk.h>
#include <stdlib.h>

const int NUM_REPEATS = 200;
const int SIZE = 100000000;

#define ITER cilk_for (int i = 0; i < SIZE; i++)
#define FUSION

int main() {
  int *A = (int*) malloc(SIZE * sizeof(int));
  int *B = (int*) malloc(SIZE * sizeof(int));
  int *C = (int*) malloc(SIZE * sizeof(int));

  for (int k = 0; k < NUM_REPEATS; k++) {
#ifdef FUSION
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
  return 0;
}
