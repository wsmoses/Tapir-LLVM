#include <cilk/cilk.h>


__attribute__((const))
double foo();


void bar();
void bat(double n);

__attribute__((const))
double moo();

__attribute__((const))
int cond();

extern double global;

void test() {
  double n;
  cilk_spawn {
    n = foo();
  }

  if (cond()) {
    cilk_sync;
  }
  global = moo();

  cilk_sync;
  bat(n);
}
