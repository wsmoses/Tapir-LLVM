#include <stdio.h>

#include <cilk/cilk.h>

#define N 100000000

__attribute__((always_inline))
int f(int x) {
    return x * x;
}

__attribute__((always_inline))
int g(int x) {
    return x * x + 3;
}

int main(void)
{
    int sum = 0;

    cilk_for (int i=0; i<N; i++) {
        sum += f(i);
    }

    cilk_for (int i=0; i<N; i++) {
        sum += g(i);
    }

    printf("%d\n", sum);
}