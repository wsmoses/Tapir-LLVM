#include <cilk/cilk.h>
#include "../common.h"

#ifndef POWER
#define POWER 10
#endif

const int SIZE = 1 << POWER;
const int NUM_REPEATS = (1ULL << 28) / SIZE;

#ifdef OPT
int sum(int *arr, int size) {
    if (size == 1) {
        return arr[0];
    }

    int result_a;
    int result_b;

    cilk_spawn {
        if (size / 2 == 1) {
            result_a = arr[0];
        }

        int a = sum(arr + 0 * (size / 4), size / 4);
        int b = sum(arr + 1 * (size / 4), size / 4);

        cilk_sync;
        result_a = a + b;
    }

    {
        if (size / 2 == 1) {
            result_b = arr[size / 2];
        }

        int a = sum(arr + 2 * (size / 4), size / 4);
        int b = sum(arr + 3 * (size / 4), size / 4);

        cilk_sync;
        result_b = a + b;
    }

    cilk_sync;
    return result_a + result_b;
}
#else
int sum(int *arr, int size) {
    if (size == 1) {
        return arr[0];
    }

    int a = cilk_spawn sum(arr + 0 * (size / 2), size / 2);
    int b =            sum(arr + 1 * (size / 2), size / 2);

    cilk_sync;
    return a + b;
}
#endif

int main(void)
{
    int *A = (int *) malloc(SIZE * sizeof(int));

    profile_start();
    int result = 0;
    for (int i=0; i<NUM_REPEATS; i++) {
        result += sum(A, SIZE);
    }
    profile_end();

    return result;
}
