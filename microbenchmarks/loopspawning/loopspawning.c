#include <cilk/cilk.h>
#include "../common.h"

#ifndef POWER
#define POWER 12
#endif

const int SIZE = (1 << POWER);
const int NUM_REPEATS = (1ULL << 28) / SIZE;
const int NUM_PARTS = 3;

int sum(int *arr, int size) {
  if (size == 1) {
    return arr[0];
  }

#ifdef OPT
  int a = cilk_spawn sum(arr + 0 * (size / 4), size / 4);
  int b = cilk_spawn sum(arr + 1 * (size / 4), size / 4);
  int c = cilk_spawn sum(arr + 2 * (size / 4), size / 4);
  int d =            sum(arr + 3 * (size / 4), size / 4);

  cilk_sync;
  return a + b + c + d;
#else
  int a = cilk_spawn sum(arr + 0 * (size / 2), size / 2);
  int b =            sum(arr + 1 * (size / 2), size / 2);

  cilk_sync;
  return a + b;
#endif
}


int main() {
  int *A = (int *) malloc(SIZE * sizeof(int));
  profile_start();
  int a;
  for (int i = 0; i < NUM_REPEATS; i++) {
    a += sum(A, SIZE);
  }
  profile_end();
  return a;
}




