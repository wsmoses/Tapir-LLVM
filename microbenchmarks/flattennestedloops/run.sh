#!/bin/bash
../../build/bin/clang -ftapir=cilk -O3 vadd.c -o vadd_nonflat
../../build/bin/clang -DFLAT -ftapir=cilk -O3 vadd.c -o vadd_flat

CORES=8

echo "===================== NON FLAT VERSION ====================="
CILK_NWORKERS=$CORES taskset -c 1-$CORES time ./vadd
echo "======================= FLAT VERSION ======================="
CILK_NWORKERS=$CORES taskset -c 1-$CORES time ./vadd_flat
