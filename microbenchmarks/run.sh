#!/bin/bash

BENCH=$1

../build/bin/clang -fcilkplus -O3 $BENCH/$BENCH.c -o "/tmp/$BENCH"
../build/bin/clang -DOPT -fcilkplus -O3 $BENCH/$BENCH.c -o "/tmp/$BENCH""_opt"

CORES=${2:-8}

echo "Running on $CORES cores"
echo "===================== NON OPT VERSION ====================="
CILK_NWORKERS=$CORES taskset -c 1-$CORES "/tmp/$BENCH"
echo "======================= OPT VERSION ======================="
CILK_NWORKERS=$CORES taskset -c 1-$CORES "/tmp/$BENCH""_opt"
